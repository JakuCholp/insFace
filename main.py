import os
import time
import threading
import json
import traceback
from urllib.parse import quote

import cv2
import numpy as np
import requests
import webbrowser
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from flask import Flask, Response, render_template_string, jsonify

# ==========================================================
# 1. КОНФИГУРАЦИЯ
# ==========================================================
CAM_IP = os.environ.get("CAM_IP", "192.168.88.70")
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASSWORD = os.environ.get("CAM_PASSWORD", "Bars1912_")

CONFIG = {
    'RTSP_URL': f"rtsp://{CAM_USER}:{quote(CAM_PASSWORD, safe='')}@{CAM_IP}:554/cam/realmonitor?channel=1&subtype=0",
    'MODEL_NAME': 'buffalo_l',
    'ROI': (0.0, 0.0, 1.0, 1.0),
    'SHOW_ROI': False,           
    
    'DET_MAX_WIDTH': 640,
    'ANALYZE_INTERVAL': 0.0,     
    'STREAM_FPS': 25,            
    'DET_SCORE_MIN': 0.45,
    'MIN_FACE_SIZE': 35,
    'BLUR_THRESHOLD': 15,
    
    'CONFIRM_FRAMES': 2,         # 2 кадра для мгновенного узнавания сотрудника
    'MIN_STRANGER_FRAMES': 3,    # 3 кадра для незнакомца (решение будет приниматься за долю секунды)
    'FAST_ACCEPT': 0.38,
    'STRICT_ENTRY': 0.38,
    'UNKNOWN_THRESHOLD': 0.38,   
    
    'HOLD_SECONDS': 1.5,         # ИСПРАВЛЕНО: Внутренняя память возвращена на 1.5 сек (защита от "амнезии")
    'DRAW_TIMEOUT': 0.2,         # Визуальная рамка исчезает быстро (не висит в воздухе)
    'MIN_IOU': 0.10,
    'OUT_WIDTH': 1280,
    'PORT': 5001,
    
    # --- НАСТРОЙКИ DJANGO API ---
    'DJANGO_API_URL': "http://127.0.0.1:8000/unknown-visitors/",       
    'DJANGO_STAFF_API_URL': "http://127.0.0.1:8000/api/entry-exit/", 
    'CAMERA_ID': 1,                                                      
    
    # --- НАСТРОЙКИ АНТИ-СПАМА ---
    'COOLDOWN_MINUTES': 3,
}

known_strangers_cache = []
cache_lock = threading.Lock()
staff_last_seen = {}
stranger_last_seen = {}

def sync_strangers_from_django():
    global known_strangers_cache
    try:
        response = requests.get(CONFIG['DJANGO_API_URL'], timeout=5)
        if response.status_code == 200:
            items = []
            for item in response.json():
                items.append({
                    "id": item.get("visitor_id", "str_0"),
                    "embedding": np.array(json.loads(item["embedding_json"]), dtype=np.float32),
                    "visits": item.get("visits_count", 1),
                })
            with cache_lock:
                known_strangers_cache = items
            print(f"📦 Загружено незнакомцев из базы Django: {len(items)}")
    except Exception as e:
        print(f"⚠️ Ошибка синхронизации с базой незнакомцев: {e}")

sync_strangers_from_django()

latest_detection = {"name": "Ожидание...", "status": "waiting", "time": "-", "match": 0, "access": "-"}

# ==========================================================
# 2. ФУНКЦИИ
# ==========================================================
def compute_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

def blur_score(crop):
    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def post_to_django(visitor_id, embedding, crop=None):
    def _send():
        try:
            data = {'visitor_id': visitor_id, 'embedding_json': json.dumps(embedding.tolist())}
            files = None
            if crop is not None:
                ok, enc = cv2.imencode('.jpg', crop)
                if ok:
                    files = {'photo': ('face.jpg', enc.tobytes(), 'image/jpeg')}
            
            res = requests.post(CONFIG['DJANGO_API_URL'], data=data, files=files, timeout=5)
            
            if res.status_code not in (200, 201):
                print(f"⚠️ Ошибка API (Незнакомец ID {visitor_id}): {res.text}")
            else:
                print(f"✅ Незнакомец успешно сохранен в базу!")
        except Exception as e:
            print(f"⚠️ Ошибка связи с сервером (Незнакомец): {e}")
    threading.Thread(target=_send, daemon=True).start()

def post_staff_to_django(staff_id, similarity):
    def _send():
        try:
            data = {
                'staff': int(staff_id), 
                'camera': CONFIG['CAMERA_ID']
            }
            res = requests.post(CONFIG['DJANGO_STAFF_API_URL'], data=data, timeout=5)
            
            if res.status_code not in (200, 201):
                print(f"⚠️ Ошибка API (Сотрудник ID {staff_id}): {res.text}")
            else:
                print(f"✅ Лог сотрудника (ID {staff_id}) успешно отправлен на сервер.")
        except Exception as e:
            print(f"⚠️ Ошибка отправки лога сотрудника: {e}")
            
    threading.Thread(target=_send, daemon=True).start()

def create_demo_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (20, 20, 25)
    cv2.putText(frame, "RTSP Stream Offline - reconnecting...", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    return frame

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

class CameraStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.running = True
        self.is_connected = False
        self.frame = create_demo_frame()
        self.frame_id = 0
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

    def update(self):
        self._connect()
        while self.running:
            if self.cap.isOpened() and self.cap.grab():
                ok, frame = self.cap.retrieve()
                if ok:
                    self.frame = frame
                    self.frame_id += 1
                    self.is_connected = True
                    continue
            self.is_connected = False
            self.frame = create_demo_frame()
            self.frame_id += 1
            time.sleep(2.0)
            self._connect()

    def read(self):
        return True, self.frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

# ==========================================================
# 3. НЕЙРОСЕТЬ
# ==========================================================
print(f"1. Загрузка нейросети ({CONFIG['MODEL_NAME']})...")
app = FaceAnalysis(name=CONFIG['MODEL_NAME'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(1280, 736))

known_embeddings = []
known_names = []
base_dir = 'known_faces'
os.makedirs(base_dir, exist_ok=True)

for person_name in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    embs = []
    for file in os.listdir(person_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                faces = app.get(img)
                if faces:
                    embs.append(faces[0].normed_embedding)
    if embs:
        mean_emb = np.mean(embs, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        known_embeddings.append(mean_emb)
        known_names.append(person_name)
print(f"👥 Сотрудников в базе: {len(known_names)}")

flask_app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8"><title>AI Security Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 0; }
        .header { background-color: #161b22; padding: 15px 30px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { margin: 0; font-size: 20px; color: #58a6ff; }
        .main-container { display: flex; padding: 20px; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .video-section { flex: 2; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px; }
        .video-section img { width: 100%; height: auto; border-radius: 5px; }
        .panel-section { flex: 1; display: flex; flex-direction: column; gap: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 25px; }
        .card-title { font-size: 14px; text-transform: uppercase; color: #8b949e; margin-bottom: 10px; letter-spacing: 1px; }
        .status-box { font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
        .status-recognized { background: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid #2ea043; }
        .status-stranger { background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid #d29922; }
        .status-waiting { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
        .data-row { display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 18px; border-bottom: 1px solid #21262d; padding-bottom: 5px; }
        .data-label { color: #8b949e; } .data-value { font-weight: bold; color: #ffffff; }
    </style>
</head>
<body>
    <div class="header"><h1>Face Recognition AI Engine</h1><div>Live RTSP Feed</div></div>
    <div class="main-container">
        <div class="video-section"><img src="{{ url_for('video_feed') }}" alt="Live Feed"></div>
        <div class="panel-section">
            <div class="card">
                <div class="card-title">Последнее распознавание</div>
                <div id="statusBox" class="status-box status-waiting">ОЖИДАНИЕ...</div>
                <div class="data-row"><span class="data-label">Личность:</span><span class="data-value" id="valName">-</span></div>
                <div class="data-row"><span class="data-label">Время:</span><span class="data-value" id="valTime">-</span></div>
                <div class="data-row"><span class="data-label">Тип доступа:</span><span class="data-value" id="valAccess">-</span></div>
                <div class="data-row"><span class="data-label">Совпадение:</span><span class="data-value" id="valMatch">0%</span></div>
            </div>
        </div>
    </div>
    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                
                document.getElementById('valName').innerText = data.name;
                document.getElementById('valTime').innerText = data.time;
                document.getElementById('valAccess').innerText = data.access;
                document.getElementById('valMatch').innerText = data.match + '%';
                const sBox = document.getElementById('statusBox');
                sBox.className = 'status-box ' + (data.status === 'recognized' ? 'status-recognized' : data.status === 'stranger' ? 'status-stranger' : 'status-waiting');
                sBox.innerText = data.status === 'recognized' ? 'СОТРУДНИК ПОДТВЕРЖДЕН' : data.status === 'stranger' ? 'УЧЕТ ПРОХОЖЕГО' : 'ОЖИДАНИЕ...';
            } catch(e) {}
        }
        setInterval(fetchStatus, 800);
    </script>
</body>
</html>
"""

def is_confident_staff(embs):
    if not known_embeddings:
        return False
    mean_emb = np.mean(embs, axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    return max(compute_similarity(e, mean_emb) for e in known_embeddings) > CONFIG['FAST_ACCEPT']


def decide_identity(track, crop):
    global latest_detection, staff_last_seen, stranger_last_seen
    
    mean_emb = np.mean(track["embs"], axis=0)
    mean_emb /= np.linalg.norm(mean_emb)

    best_sim, best_name = 0.0, None
    for emb, name in zip(known_embeddings, known_names):
        sim = compute_similarity(emb, mean_emb)
        if sim > best_sim:
            best_sim, best_name = sim, name

    now_str = time.strftime("%H:%M:%S")
    now_ts = time.time()

    # === ЛОГИКА ДЛЯ СОТРУДНИКОВ ===
    if best_name and best_sim > CONFIG['STRICT_ENTRY']:
        if '_' in best_name:
            staff_id_str, staff_display_name = best_name.split('_', 1)
        else:
            staff_id_str, staff_display_name = best_name, best_name

        track.update(kind="staff", label=staff_display_name.replace("_", " "), staff_id=staff_id_str)
        latest_detection = {"name": staff_display_name.replace("_", " "), "status": "recognized",
                            "time": now_str, "match": int(best_sim * 100), "access": "Сотрудник"}
        
        last_seen = staff_last_seen.get(staff_id_str, 0)
        cooldown_seconds = CONFIG['COOLDOWN_MINUTES'] * 60
        
        if (now_ts - last_seen) > cooldown_seconds:
            post_staff_to_django(staff_id_str, best_sim)
            
        staff_last_seen[staff_id_str] = now_ts
        return

    # === ЛОГИКА ДЛЯ ПРОХОЖИХ (НЕЗНАКОМЦЕВ) ===
    if len(track["embs"]) < CONFIG['MIN_STRANGER_FRAMES']:
        return

    matched = None
    with cache_lock:
        best_s = 0.0
        for s in known_strangers_cache:
            sim = compute_similarity(s["embedding"], mean_emb)
            if sim > CONFIG['UNKNOWN_THRESHOLD'] and sim > best_s:
                best_s, matched = sim, s
        
        cooldown_seconds = CONFIG['COOLDOWN_MINUTES'] * 60
        
        if matched:
            vid = matched["id"]
            last_seen = stranger_last_seen.get(vid, 0)
            
            updated_emb = matched["embedding"] * 0.8 + mean_emb * 0.2
            updated_emb /= np.linalg.norm(updated_emb)
            matched["embedding"] = updated_emb
            
            if (now_ts - last_seen) > cooldown_seconds:
                matched["visits"] += 1
                new_visit = True
                post_to_django(vid, updated_emb, crop=None)
            else:
                new_visit = False
                
            label = f"Visitor (Visits: {matched['visits']})"
        else:
            # Используем миллисекунды для генерации уникального ID незнакомца
            vid = f"str_{int(time.time() * 1000)}"
            label = "New Visitor"
            known_strangers_cache.append({"id": vid, "embedding": mean_emb, "visits": 1})
            new_visit = True
            post_to_django(vid, mean_emb, crop)

    track.update(kind="stranger", label=label, stranger_id=vid)
    latest_detection = {"name": "Незнакомец", "status": "stranger", "time": now_str,
                        "match": int(best_sim * 100) if best_sim else 0, "access": "Учет прохожего"}

    stranger_last_seen[vid] = now_ts


camera = None
tracks = {}
tracks_lock = threading.Lock()
analysis_size = (0, 0)

def ceil32(x):
    return int(((x + 31) // 32) * 32)

def get_roi_rect(frame_shape):
    h, w = frame_shape[:2]
    fx1, fy1, fx2, fy2 = CONFIG['ROI']
    return int(w * fx1), int(h * fy1), int(w * fx2), int(h * fy2)

def analysis_loop():
    global latest_detection, tracks, staff_last_seen, stranger_last_seen
    detector = app.models['detection']
    recognizer = app.models['recognition']
    next_id = 0
    last_id = -1
    prepared_for = None

    while True:
        try:
            if not camera.is_connected or camera.frame_id == last_id:
                time.sleep(0.01)
                continue
            
            started = time.time()
            last_id = camera.frame_id
            frame = camera.frame
            now = started

            rx1, ry1, rx2, ry2 = get_roi_rect(frame.shape)
            if prepared_for != (rx1, ry1, rx2, ry2):
                roi_w, roi_h = rx2 - rx1, ry2 - ry1
                scale = min(1.0, CONFIG['DET_MAX_WIDTH'] / roi_w)
                det_size = (ceil32(roi_w * scale), ceil32(roi_h * scale))
                app.prepare(ctx_id=0, det_size=det_size)
                prepared_for = (rx1, ry1, rx2, ry2)

            roi_img = np.ascontiguousarray(frame[ry1:ry2, rx1:rx2])
            bboxes, kpss = detector.detect(roi_img, max_num=0, metric='default')

            with tracks_lock:
                for k in [k for k, v in tracks.items() if now - v["time"] > CONFIG['HOLD_SECONDS']]:
                    del tracks[k]

                if bboxes is not None:
                    for i in range(len(bboxes)):
                        score = float(bboxes[i, 4])
                        if score < CONFIG['DET_SCORE_MIN']:
                            continue
                        x1, y1, x2, y2 = (bboxes[i, :4] + np.array([rx1, ry1, rx1, ry1])).astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        box = (x1, y1, x2, y2)
                        
                        if (x2 - x1) < CONFIG['MIN_FACE_SIZE'] or (y2 - y1) < CONFIG['MIN_FACE_SIZE']:
                            continue
                        
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        tid, best_iou = None, CONFIG['MIN_IOU']
                        for k, t in tracks.items():
                            iou = compute_iou(box, t["box"])
                            if iou > best_iou:
                                tid, best_iou = k, iou

                        sharp = blur_score(crop) >= CONFIG['BLUR_THRESHOLD']

                        if tid is None:
                            if not sharp:
                                continue
                            tid = next_id
                            next_id += 1
                            tracks[tid] = {"box": box, "time": now, "embs": [], "kind": None, "label": None}

                        t = tracks[tid]
                        t["box"], t["time"] = box, now

                        if t.get("kind") == "staff" and "staff_id" in t:
                            staff_last_seen[t["staff_id"]] = now
                        if t.get("kind") == "stranger" and "stranger_id" in t:
                            stranger_last_seen[t["stranger_id"]] = now

                        if t["kind"] is None and sharp and kpss is not None:
                            face = Face(bbox=bboxes[i, :4] + np.array([rx1, ry1, rx1, ry1]),
                                        kps=kpss[i] + np.array([rx1, ry1]), det_score=score)
                            recognizer.get(frame, face)
                            t["embs"].append(face.normed_embedding)
                            
                            if len(t["embs"]) >= CONFIG['CONFIRM_FRAMES'] or is_confident_staff(t["embs"]):
                                decide_identity(t, crop.copy())

                if not tracks:
                    latest_detection = {"name": "-", "status": "waiting", "time": "-", "match": 0, "access": "-"}

            spent = time.time() - started
            if spent < CONFIG['ANALYZE_INTERVAL']:
                time.sleep(CONFIG['ANALYZE_INTERVAL'] - spent)

        except Exception as e:
            print(f"⚠️ Ошибка в главном потоке: {e}")
            traceback.print_exc()
            time.sleep(1)

def generate_frames():
    last_sent = -1
    frame_delay = 1.0 / CONFIG['STREAM_FPS']
    while True:
        started = time.time()
        if camera.frame_id == last_sent:
            time.sleep(0.005)
            continue
        last_sent = camera.frame_id
        frame = camera.frame.copy()

        if camera.is_connected:
            if CONFIG['SHOW_ROI']:
                rx1, ry1, rx2, ry2 = get_roi_rect(frame.shape)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 1)
            
            with tracks_lock:
                now = time.time()
                snapshot = [dict(t) for t in tracks.values() if now - t["time"] < CONFIG['DRAW_TIMEOUT']]
                
            for t in snapshot:
                x1, y1, x2, y2 = t["box"]
                if t["kind"] == "staff":
                    color, text = (0, 255, 0), t["label"]
                elif t["kind"] == "stranger":
                    color, text = (0, 165, 255), t["label"]
                else:
                    color = (160, 160, 160)
                    text = "" 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        h, w = frame.shape[:2]
        if w > CONFIG['OUT_WIDTH']:
            frame = cv2.resize(frame, (CONFIG['OUT_WIDTH'], int(h * CONFIG['OUT_WIDTH'] / w)), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        spare = frame_delay - (time.time() - started)
        if spare > 0:
            time.sleep(spare)

@flask_app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@flask_app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route('/status')
def get_status():
    return jsonify(latest_detection)

if __name__ == '__main__':
    camera = CameraStream(CONFIG['RTSP_URL'])
    threading.Thread(target=analysis_loop, daemon=True).start()
    url = f"http://127.0.0.1:{CONFIG['PORT']}"
    print(f"🌐 Панель запущена: {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    flask_app.run(host='0.0.0.0', port=CONFIG['PORT'], debug=False, use_reloader=False, threaded=True)