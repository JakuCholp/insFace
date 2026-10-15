import os
import re
import copy
import time
import platform
import subprocess
import threading
import json
import traceback
from urllib.parse import quote

import cv2
import numpy as np
import onnxruntime as ort
import requests
import webbrowser
from insightface.app import FaceAnalysis
from insightface.utils import face_align
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

    'DET_MAX_WIDTH': 'auto',     # Ширина кадра для детектора: 'auto' ≈ ширина/3 (640 для 1080p, 864 для 4MP). Больше — дальние лица, но медленнее
    'REFINE_KPS': True,          # Уточнять точки глаз/носа на кропе в полном разрешении (точнее выравнивание лица)...
    'REFINE_MAX_DET_PX': 40,     # ...но только мелким лицам (меньше стольких px во входе детектора, ~120 px в 1080p)
    'MAX_EMBEDS_PER_FRAME': 'auto',  # Сколько лиц за кадр распознавать (новые — первыми). 'auto': 2 на CPU, 4 с ускорителем
    'PROVIDER': os.environ.get('FACE_PROVIDER', 'auto'),  # На чём считать нейросеть: auto | cpu | coreml | cuda
    'ORT_THREADS': int(os.environ.get('ORT_THREADS', '0')),  # Потоков CPU для нейросети; 0 = подобрать (на M1/M2 — только P-ядра)
    'ANALYZE_INTERVAL': 0.0,
    'STREAM_FPS': 15,            # Видео на панели: кодируется один раз для всех вкладок
    'DET_SCORE_MIN': 0.45,
    'MIN_FACE_SIZE': 35,         # Лица меньше этого не трекаем вообще
    'MIN_REC_FACE_SIZE': 50,     # Лица меньше этого трекаем, но не распознаём: эмбеддинг будет мусорным
    'MAX_YAW': 0.8,              # Поворот головы, после которого кадр выбрасывается (0 = анфас, ~0.5 = 35-40°, 1 = профиль)
    'BLUR_THRESHOLD': 15,        # Совсем смазанные кадры выбрасываются
    'STRANGER_MIN_BLUR': 25,     # Кадр мутнее этого не доказывает "не сотрудник": на нём и своего не узнать

    # --- ПОРОГИ РАСПОЗНАВАНИЯ ---
    'STAFF_THRESHOLD': 0.38,     # Сходство среднего лица трека с лучшим фото сотрудника
    'FAST_ACCEPT': 0.48,         # Сходство одного кадра, после которого сотрудник узнаётся сразу
    'STAFF_GRAY_ZONE': 0.28,     # Похож на сотрудника "не до конца" — не спешим записывать в незнакомцы
    'STRANGER_THRESHOLD': 0.38,  # Сходство, при котором незнакомец считается уже виденным
    'CONFIRM_FRAMES': 2,         # Кадров для подтверждения сотрудника по среднему
    'MIN_STRANGER_FRAMES': 3,    # Чётких кадров, прежде чем признать человека незнакомцем
    'MIN_STRANGER_QUALITY': 1.2, # Суммарное качество этих кадров
    'EXIT_MIN_FRAMES': 2,        # Человек быстро ушёл: решаем по стольким чётким кадрам, а не выбрасываем трек
    'EXIT_MIN_QUALITY': 0.7,
    'MAX_DECISION_FRAMES': 8,    # В "серой зоне" ждём максимум столько кадров
    'TRACK_MAX_EMBS': 15,        # Сколько лучших кадров трека хранить
    'STRANGER_GALLERY_SIZE': 8,  # Сколько разных ракурсов помнить на одного незнакомца
    'STRANGER_COMMIT_SECONDS': 3.0,  # Сколько секунд копить кадры незнакомца перед отправкой в базу

    'HOLD_SECONDS': 1.5,         # Внутренняя память трека (защита от "амнезии")
    'REVERIFY_GAP': 0.5,         # Если трек не видели дольше — новое лицо сверяем с ним по эмбеддингу
    'DRAW_TIMEOUT': 0.3,         # Минимум; реально подстраивается под скорость анализа
    'MIN_IOU': 0.10,
    'OUT_WIDTH': 1280,
    'PORT': 5001,
    'DEBUG_LOG': True,           # Печатать сходства при каждом решении — по ним удобно подбирать пороги
    'PERF_LOG_SECONDS': 15,      # Раз в столько секунд печатать скорость анализа (0 — не печатать)

    # --- НАСТРОЙКИ DJANGO API ---
    'DJANGO_API_URL': "http://127.0.0.1:8000/unknown-visitors/",
    'DJANGO_STAFF_API_URL': "http://127.0.0.1:8000/api/entry-exit/",
    'CAMERA_ID': 1,
    'STRANGER_SYNC_MINUTES': 5,  # Как часто подтягивать незнакомцев из базы (например, с других камер)

    # --- НАСТРОЙКИ АНТИ-СПАМА ---
    'COOLDOWN_MINUTES': 3,
}

EMB_DIM = 512
known_strangers = {}         # visitor_id -> {"id", "gallery": np.ndarray (k, 512), "visits"}
cache_lock = threading.Lock()
staff_last_seen = {}
stranger_last_seen = {}


def parse_embedding(raw):
    """embedding_json из Django может прийти строкой JSON или уже списком (JSONField).
    Возвращает массив (k, 512) нормированных эмбеддингов или None."""
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != EMB_DIM:
        return None
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr[norms[:, 0] > 0] / norms[norms[:, 0] > 0]
    return arr if len(arr) else None


def fetch_strangers_from_django():
    url, items, pages = CONFIG['DJANGO_API_URL'], [], 0
    while url and pages < 1000:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):  # DRF с пагинацией: {"count", "next", "results"}
            items.extend(data.get("results", []))
            url = data.get("next")
        else:
            items.extend(data)
            url = None
        pages += 1
    return items


def sync_strangers_from_django(verbose=True):
    try:
        raw_items = fetch_strangers_from_django()
    except Exception as e:
        print(f"⚠️ Ошибка синхронизации с базой незнакомцев: {e}")
        return

    loaded, skipped = {}, 0
    for item in raw_items:
        # Одна битая запись больше не обнуляет весь кэш
        try:
            vid = item.get("visitor_id") or item.get("id")
            gallery = parse_embedding(item.get("embedding_json"))
            visits = int(item.get("visits_count") or 1)
        except Exception:
            vid, gallery = None, None
        if vid is None or gallery is None:
            skipped += 1
            continue
        loaded[str(vid)] = {"id": str(vid), "gallery": gallery, "visits": visits}

    added = 0
    with cache_lock:
        for vid, s in loaded.items():
            local = known_strangers.get(vid)
            if local is None:
                known_strangers[vid] = s
                added += 1
            else:
                # Локальная галерея богаче (несколько ракурсов) — оставляем её, обновляем только счётчик
                local["visits"] = max(local["visits"], s["visits"])
        total = len(known_strangers)

    if verbose or added or skipped:
        print(f"📦 Незнакомцев в базе Django: {len(loaded)}, новых: {added}, всего в памяти: {total}"
              + (f", пропущено битых записей: {skipped}" if skipped else ""))
    if skipped and not loaded:
        print("   ⚠️ Ни одной записи не удалось разобрать. Проверь, что API отдаёт поля visitor_id и embedding_json.")


def stranger_sync_loop():
    while True:
        time.sleep(CONFIG['STRANGER_SYNC_MINUTES'] * 60)
        sync_strangers_from_django(verbose=False)


sync_strangers_from_django()

latest_detection = {"name": "Ожидание...", "status": "waiting", "time": "-", "match": 0, "access": "-"}
latest_decision_at = 0.0     # Когда на панели появилось последнее решение
last_vid_ms = 0              # Последний выданный ID незнакомца (миллисекунды)
analysis_period = 0.1

# ==========================================================
# 2. ФУНКЦИИ
# ==========================================================
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

def center_distance(a, b):
    """Расстояние между центрами рамок в долях размера рамки b. inf — если размеры слишком разные."""
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    if not 0.5 < (wa * ha) / max(1, wb * hb) < 2.0:
        return float('inf')
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return float(np.hypot(dx, dy)) / max(1, wb, hb)

def blur_score(aligned):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_yaw(kps):
    """0 = анфас, ~0.5 = поворот на 35-40°, 1 = профиль. Смещение носа от середины глаз вдоль линии глаз."""
    le, re_, nose = kps[0], kps[1], kps[2]
    eye_vec = re_ - le
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-3:
        return 1.0
    offset = float(np.dot(nose - (le + re_) / 2, eye_vec / eye_dist))
    return min(1.0, abs(offset) / eye_dist)

def face_crop(frame, box, margin=0.25):
    x1, y1, x2, y2 = box
    m = int(margin * max(x2 - x1, y2 - y1))
    h, w = frame.shape[:2]
    return frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)].copy()

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
                print(f"⚠️ Ошибка API (Незнакомец ID {visitor_id}): {res.status_code} {res.text[:300]}")
            else:
                print(f"✅ Незнакомец {visitor_id} сохранен в базу!")
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
                print(f"⚠️ Ошибка API (Сотрудник ID {staff_id}): {res.status_code} {res.text[:300]}")
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
# 3. НЕЙРОСЕТЬ: УСКОРИТЕЛЬ (Neural Engine / GPU) И НАСТРОЙКА CPU
# ==========================================================
print(f"1. Загрузка нейросети ({CONFIG['MODEL_NAME']})...")
# Базовые сессии — на CPU (по умолчанию insightface просит CUDA, которой на Mac нет). Ускоритель подбирается ниже.
app = FaceAnalysis(name=CONFIG['MODEL_NAME'], allowed_modules=['detection', 'recognition'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 736), det_thresh=CONFIG['DET_SCORE_MIN'])  # только для фото сотрудников
detector = app.models['detection']
recognizer = app.models['recognition']
REFINE_INPUT = (224, 224)


def perf_cores():
    """Число производительных ядер на Apple Silicon: энергоэффективные ядра только тормозят нейросеть."""
    if platform.system() != 'Darwin':
        return None
    try:
        out = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) or None
    except Exception:
        return None

def session_options(threads=0):
    so = ort.SessionOptions()  # новый объект на каждую сессию: общий на несколько CoreML-сессий может уронить ORT
    so.log_severity_level = 3
    # Иначе потоки onnxruntime после каждого вызова крутятся вхолостую и отнимают CPU у камеры и видеопотока
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    if threads:
        so.intra_op_num_threads = threads
    return so

def static_model_bytes(path, hw=None):
    """Копия модели с фиксированным размером входа: CoreML не любит "плавающие" размеры."""
    import onnx
    m = onnx.load(path)
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[0].ClearField('dim_param')
    dims[0].dim_value = 1
    if hw is not None:
        for d, v in zip(dims[2:4], hw):
            d.ClearField('dim_param')
            d.dim_value = int(v)
    for o in m.graph.output:
        o.type.tensor_type.ClearField('shape')
    del m.graph.value_info[:]
    return m.SerializeToString()

class Batch1Session:
    """Сессия с фиксированным батчем 1, которой всё равно можно подать несколько лиц сразу."""
    def __init__(self, sess):
        self.sess = sess

    def run(self, output_names, feed):
        (name, x), = feed.items()
        outs = [self.sess.run(output_names, {name: x[i:i + 1]}) for i in range(x.shape[0])]
        return [np.concatenate([o[k] for o in outs], axis=0) for k in range(len(outs[0]))]

    def get_providers(self):
        return self.sess.get_providers()

    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

def accel_candidates():
    want = CONFIG['PROVIDER'].lower()
    avail = ort.get_available_providers()
    out = []
    if want in ('auto', 'cuda') and 'CUDAExecutionProvider' in avail:
        out.append(('CUDA', [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}),
                             'CPUExecutionProvider'], False))
    if want in ('auto', 'coreml') and 'CoreMLExecutionProvider' in avail:
        ort_version = tuple(int(x) for x in re.findall(r'\d+', ort.__version__)[:2])
        if ort_version >= (1, 21):
            for fmt, units in (('MLProgram', 'ALL'), ('MLProgram', 'CPUAndGPU'), ('NeuralNetwork', 'ALL')):
                opts = {'ModelFormat': fmt, 'MLComputeUnits': units, 'RequireStaticInputShapes': '1'}
                out.append((f'CoreML {fmt}/{units}', [('CoreMLExecutionProvider', opts), 'CPUExecutionProvider'], True))
        else:  # старый onnxruntime молча игнорирует эти опции — пробовать варианты бессмысленно
            out.append(('CoreML', ['CoreMLExecutionProvider', 'CPUExecutionProvider'], True))
    return out

def run_each(sess, x):
    """По одному образцу: статические CoreML-модели принимают только батч 1."""
    name = sess.get_inputs()[0].name
    return [sess.run(None, {name: x[i:i + 1]}) for i in range(len(x))]

def median_time(sess, x, n=7):
    feed = {sess.get_inputs()[0].name: x[:1]}
    for _ in range(2):
        sess.run(None, feed)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def outputs_agree(kind, ref, out):
    if not all(np.isfinite(o).all() for b in out for o in b):
        return False, "выдаёт NaN/inf"
    if kind == 'rec':
        cos = min(float(np.dot(a[0].ravel(), b[0].ravel()) /
                        (np.linalg.norm(a[0]) * np.linalg.norm(b[0]) + 1e-12)) for a, b in zip(ref, out))
        return cos >= 0.995, f"совпадение с CPU {cos:.4f}"
    # У детектора выходы: 3 карты уверенности, затем рамки и точки лица для шагов 8/16/32 (в единицах шага).
    # Точки важны не меньше уверенности — по ним выравнивается лицо перед распознаванием.
    diff_score, diff_px = 0.0, 0.0
    for a, b in zip(ref, out):
        for i, stride in enumerate((8, 16, 32)):
            diff_score = max(diff_score, float(np.abs(a[i] - b[i]).max()))
            mask = a[i].reshape(-1) > 0.3
            for head in (i + 3, i + 6):
                if mask.any() and head < len(a):
                    d = np.abs(a[head].reshape(mask.size, -1)[mask] - b[head].reshape(mask.size, -1)[mask]).max()
                    diff_px = max(diff_px, float(d) * stride)
    return diff_score <= 0.02 and diff_px <= 1.0, f"расхождение с CPU: уверенность {diff_score:.3f}, точки {diff_px:.2f} px"

def best_session(model_path, kind, x, hw=None):
    """Самый быстрый ПРОВЕРЕННЫЙ способ запускать модель. Ускоритель берётся, только если он заметно быстрее
    CPU и даёт те же результаты; при любой ошибке остаёмся на CPU (принудительно: FACE_PROVIDER=cpu)."""
    pc = perf_cores()
    threads = [CONFIG['ORT_THREADS']] if CONFIG['ORT_THREADS'] else [0] + ([pc] if pc else [])
    best = None
    for th in threads:
        sess = ort.InferenceSession(model_path, session_options(th), providers=['CPUExecutionProvider'])
        ms = median_time(sess, x)
        if best is None or ms < best[1]:
            best = (sess, ms, f"CPU, потоков: {th or 'авто'}")
    cpu_sess, cpu_ms, label = best
    chosen, chosen_ms = cpu_sess, cpu_ms
    ref = run_each(cpu_sess, x)
    static_bytes, deadline = None, time.time() + 120
    for name, providers, static in accel_candidates():
        if time.time() > deadline:
            break
        try:
            if static and static_bytes is None:
                static_bytes = static_model_bytes(model_path, hw)
            sess = ort.InferenceSession(static_bytes if static else model_path, session_options(), providers=providers)
            if sess.get_providers()[0] == 'CPUExecutionProvider':
                continue
            ms = median_time(sess, x)
            ok, how = outputs_agree(kind, ref, run_each(sess, x))
            print(f"   ⚙️ {kind} {name}: {ms * 1000:.1f} мс против CPU {cpu_ms * 1000:.1f} мс, {how}")
            if ok and ms * 1.15 < cpu_ms:
                chosen, chosen_ms, label = (Batch1Session(sess) if static and kind == 'rec' else sess), ms, name
                break
        except Exception as e:
            print(f"   ⚙️ {kind} {name}: не подошёл ({type(e).__name__}: {str(e)[:100]})")
    return chosen, label, chosen_ms * 1000, cpu_ms * 1000

def test_image():
    """Настоящие лица для самопроверки ускорителя: на случайном шуме сравнивать эмбеддинги бессмысленно."""
    try:
        from insightface.data import get_image
        return get_image('t1')
    except Exception:
        return None

def rec_test_input():
    img = test_image()
    aligned = []
    if img is not None:
        aligned = [face_align.norm_crop(img, landmark=f.kps, image_size=112) for f in app.get(img)[:4]]
    if not aligned:
        rng = np.random.default_rng(0)
        aligned = [rng.integers(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(2)]
    return cv2.dnn.blobFromImages(aligned, 1.0 / recognizer.input_std, recognizer.input_size,
                                  (recognizer.input_mean,) * 3, swapRB=True)

def det_test_input(w, h):
    img = test_image()
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.dnn.blobFromImage(cv2.resize(img, (w, h)), 1.0 / detector.input_std, (w, h),
                                 (detector.input_mean,) * 3, swapRB=True)

def make_fixed_detector(det_size):
    """Отдельный детектор под один размер входа. Раньше в цикле вызывался app.prepare(det_size=...), но
    insightface 0.7.3 молча игнорирует повторный размер ("det_size is already set in detection model, ignore"),
    и детектор всё время работал на 1280x736 — в ~4 раза медленнее задуманного."""
    w, h = det_size
    sess, label, ms, cpu_ms = best_session(detector.model_file, 'det', det_test_input(w, h), hw=(h, w))
    d = copy.copy(detector)
    if hasattr(d, 'static_shape_sessions'):
        d.static_shape_sessions = False  # insightface 2.x: иначе он строит свою сессию и не использует проверенную
    d.session, d.input_size, d.center_cache = sess, (w, h), {}
    if hasattr(d, 'input_sizes'):
        d.input_sizes = [(w, h)]
    d.det_thresh = CONFIG['DET_SCORE_MIN']
    print(f"   ⚙️ Детектор {w}x{h}: {label}, {ms:.0f} мс" + (f" (CPU {cpu_ms:.0f} мс)" if ms < cpu_ms else ""))
    return d


# Распознавание подбирается ДО загрузки сотрудников: эталоны и живые кадры должны считаться одинаково
recognizer.session, rec_backend, rec_ms, rec_cpu_ms = best_session(recognizer.model_file, 'rec', rec_test_input())
print(f"⚙️ Распознавание лиц: {rec_backend}, {rec_ms:.0f} мс на лицо" +
      (f" (на CPU было бы {rec_cpu_ms:.0f} мс)" if rec_ms < rec_cpu_ms else ""))
EMBEDS_PER_FRAME = (CONFIG['MAX_EMBEDS_PER_FRAME'] if CONFIG['MAX_EMBEDS_PER_FRAME'] != 'auto'
                    else (2 if rec_backend.startswith('CPU') else 4))
refine_detector = make_fixed_detector(REFINE_INPUT)

# ==========================================================
# 3.1 БАЗА СОТРУДНИКОВ
# ==========================================================


def parse_staff_folder(folder):
    """'12_Ivan_Petrov' -> ('12', 'Ivan Petrov'). Без числового ID лог в Django отправить нельзя."""
    m = re.match(r'^(\d+)(?:[_\- ]+(.+))?$', folder)
    if m:
        return m.group(1), (m.group(2) or m.group(1)).replace('_', ' ')
    return None, folder.replace('_', ' ')

def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

def enroll_embedding(img):
    faces = app.get(img)
    if not faces:
        # Фото, обрезанное вплотную по лицу, детектор часто не видит — добавляем поля и пробуем ещё раз
        pad = int(0.4 * max(img.shape[:2]))
        faces = app.get(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    if not faces:
        return None, 0
    # Раньше брался faces[0] — это не обязательно сотрудник, а может быть кто-то на фоне
    return largest_face(faces).normed_embedding, len(faces)

def load_staff(base_dir='known_faces'):
    os.makedirs(base_dir, exist_ok=True)
    people, templates, owners, no_id = [], [], [], []

    for folder in sorted(os.listdir(base_dir)):
        person_dir = os.path.join(base_dir, folder)
        if not os.path.isdir(person_dir):
            continue
        embs, files = [], []
        for file in sorted(os.listdir(person_dir)):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(person_dir, file))
            if img is None:
                continue
            emb, n_faces = enroll_embedding(img)
            if emb is None:
                print(f"   ⚠️ {folder}/{file}: лицо не найдено, фото пропущено")
                continue
            if n_faces > 1:
                print(f"   ℹ️ {folder}/{file}: на фото {n_faces} лица, взято самое крупное")
            embs.append(emb)
            files.append(file)
        if not embs:
            print(f"   ⚠️ {folder}: ни одного пригодного фото, сотрудник не загружен")
            continue

        embs = np.asarray(embs, dtype=np.float32)
        # Фото, не похожее на остальные фото этого же человека, — скорее всего ошибка в базе
        if len(embs) >= 2:
            for i, file in enumerate(files):
                others = np.delete(embs, i, axis=0).mean(axis=0)
                sim = float(np.dot(embs[i], others / np.linalg.norm(others)))
                if sim < 0.3:
                    print(f"   ⚠️ {folder}/{file}: мало похоже на остальные фото ({sim:.2f}) — проверь, тот ли это человек")

        staff_id, name = parse_staff_folder(folder)
        if staff_id is None:
            no_id.append(folder)
        idx = len(people)
        people.append({"id": staff_id, "name": name, "folder": folder})

        # Сравниваем и с каждым фото отдельно, и с усреднённым лицом: так лучше работают разные ракурсы
        centroid = embs.mean(axis=0)
        person_templates = list(embs) + ([centroid / np.linalg.norm(centroid)] if len(embs) > 1 else [])
        templates.extend(person_templates)
        owners.extend([idx] * len(person_templates))
        print(f"   ✔ {folder}: {len(embs)} фото")

    print(f"👥 Сотрудников в базе: {len(people)}")
    if not people:
        print("   ⚠️ Папка known_faces пуста — все люди будут считаться незнакомцами!")
    if no_id:
        shown = ', '.join(no_id[:10]) + (f" и ещё {len(no_id) - 10}" if len(no_id) > 10 else "")
        print(f"   ⚠️ Нет числового ID в имени папки (нужно 'ID_Имя', например '12_Anas'), "
              f"лог входа для них в Django не отправится: {shown}")

    matrix = np.asarray(templates, dtype=np.float32).reshape(-1, EMB_DIM)
    return people, matrix, np.asarray(owners, dtype=int)


staff_people, staff_matrix, staff_owner = load_staff()

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
        .status-familiar { background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid #58a6ff; }
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
        const STATUS = {
            recognized: ['status-recognized', 'СОТРУДНИК ПОДТВЕРЖДЕН'],
            familiar: ['status-familiar', 'ЗНАКОМЫЙ НЕЗНАКОМЕЦ'],
            stranger: ['status-stranger', 'НОВЫЙ НЕЗНАКОМЕЦ'],
        };
        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();

                document.getElementById('valName').innerText = data.name;
                document.getElementById('valTime').innerText = data.time;
                document.getElementById('valAccess').innerText = data.access;
                document.getElementById('valMatch').innerText = data.match + '%';
                const [cls, text] = STATUS[data.status] || ['status-waiting', 'ОЖИДАНИЕ...'];
                const sBox = document.getElementById('statusBox');
                sBox.className = 'status-box ' + cls;
                sBox.innerText = text;
            } catch(e) {}
        }
        setInterval(fetchStatus, 800);
    </script>
</body>
</html>
"""

# ==========================================================
# 4. КАЧЕСТВО ЛИЦА И ЭМБЕДДИНГИ
# ==========================================================
def refine_landmarks(frame, box, kps, score):
    """Детектор работает на уменьшенном кадре, и у небольших лиц точки глаз/носа получаются грубыми.
    Повторно ищем лицо на кропе в полном разрешении — выравнивание точнее, эмбеддинг стабильнее."""
    x1, y1, x2, y2 = box
    m = int(0.6 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
    cx2, cy2 = min(frame.shape[1], x2 + m), min(frame.shape[0], y2 + m)
    crop = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2])
    bboxes, kpss = refine_detector.detect(crop, input_size=REFINE_INPUT, max_num=0, metric='default')
    if bboxes is None or len(bboxes) == 0 or kpss is None:
        return kps, score
    shift = np.array([cx1, cy1, cx1, cy1])
    best_i, best_iou = None, 0.3
    for i in range(len(bboxes)):
        iou = compute_iou(box, bboxes[i, :4] + shift)
        if iou > best_iou:
            best_i, best_iou = i, iou
    if best_i is None:
        return kps, score
    return kpss[best_i] + np.array([cx1, cy1]), float(bboxes[best_i, 4])

def prepare_sample(frame, det, det_scale):
    """Выравнивание и оценка качества одного лица (сама нейросеть — в embed_samples). None — если кадр
    не годится для распознавания (слишком мелкое лицо, профиль, смаз): такие кадры портят среднее лицо трека."""
    x1, y1, x2, y2 = det["box"]
    size = min(x2 - x1, y2 - y1)
    kps, score = det["kps"], det["score"]
    if kps is None or size < CONFIG['MIN_REC_FACE_SIZE']:
        return None
    # Крупным лицам уточнение почти ничего не даёт, а стоит как треть распознавания
    if CONFIG['REFINE_KPS'] and size * det_scale < CONFIG['REFINE_MAX_DET_PX']:
        if estimate_yaw(kps) > CONFIG['MAX_YAW'] + 0.15:
            return None  # явный профиль даже по грубым точкам — не тратим время на уточнение
        kps, score = refine_landmarks(frame, det["box"], kps, score)
    yaw = estimate_yaw(kps)
    if yaw > CONFIG['MAX_YAW']:
        return None
    aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
    blur = blur_score(aligned)
    if blur < CONFIG['BLUR_THRESHOLD']:
        return None
    quality = max(1e-3, score * min(1.0, size / 112.0) * (1.0 - yaw ** 2))
    # Смазанный кадр почти не несёт личности: в среднем лице трека он весит меньше
    sharpness = float(np.clip((blur - 10.0) / 30.0, 0.1, 1.0))
    return {"aligned": aligned, "emb": None, "q": quality, "w": quality * sharpness, "blur": blur,
            "crop": face_crop(frame, det["box"])}

def embed_samples(samples):
    """Все отобранные лица кадра — одним вызовом нейросети."""
    if not samples:
        return
    feats = recognizer.get_feat([s["aligned"] for s in samples])
    for s, f in zip(samples, feats):
        norm = np.linalg.norm(f)
        s["emb"] = f / norm if norm > 0 else None
        del s["aligned"]

# ==========================================================
# 5. ПРИНЯТИЕ РЕШЕНИЙ
# ==========================================================
def track_template(t):
    """Среднее лицо трека, взвешенное по качеству кадров."""
    w = np.asarray(t["weights"], dtype=np.float32)[:, None]
    emb = (np.asarray(t["embs"]) * w).sum(axis=0)
    return emb / np.linalg.norm(emb)

def best_staff_match(emb):
    if len(staff_matrix) == 0:
        return 0.0, None
    sims = staff_matrix @ emb
    i = int(np.argmax(sims))
    return float(sims[i]), staff_people[staff_owner[i]]

def best_stranger_match(emb):
    """Вызывать под cache_lock. Сравниваем со всеми запомненными ракурсами каждого незнакомца."""
    best_sim, best = 0.0, None
    for s in known_strangers.values():
        sim = float(np.max(s["gallery"] @ emb))
        if sim > best_sim:
            best_sim, best = sim, s
    return best_sim, best

def gallery_centroid(s):
    c = s["gallery"].mean(axis=0)
    return c / np.linalg.norm(c)

def add_to_gallery(s, emb):
    """Вызывать под cache_lock. Вместо старого смешивания 0.8/0.2 (эталон "уплывал" и переставал
    совпадать) храним несколько разных ракурсов человека."""
    if float(np.max(s["gallery"] @ emb)) > 0.9:
        return  # почти такой же ракурс уже есть
    s["gallery"] = np.vstack([s["gallery"], emb[None, :]])
    if len(s["gallery"]) > CONFIG['STRANGER_GALLERY_SIZE']:
        s["gallery"] = np.delete(s["gallery"], 1, axis=0)  # самый первый эталон не трогаем

def log_decision(t, text):
    if CONFIG['DEBUG_LOG']:
        blurs = sorted(t["blurs"]) or [0.0]
        n_sharp = sum(b >= CONFIG['STRANGER_MIN_BLUR'] for b in t["blurs"])
        print(f"🔎 Трек #{t['id']}: кадров {t['samples']} (чётких {n_sharp}), качество {sum(t['qs']):.1f}, "
              f"резкость {blurs[0]:.0f}..{blurs[-1]:.0f} → {text}")

def mark_staff(t, staff, sim, now):
    global latest_detection, latest_decision_at
    latest_decision_at = now
    if t["pending"] is not None:
        cancel_stranger(t)  # первые кадры ошибочно приняли сотрудника за незнакомца — в базу он не попадёт
    key = staff["id"] or staff["folder"]
    t.update(kind="staff", label=staff["name"], staff_key=key)
    latest_detection = {"name": staff["name"], "status": "recognized",
                        "time": time.strftime("%H:%M:%S"), "match": int(sim * 100), "access": "Сотрудник"}
    log_decision(t, f"сотрудник {staff['folder']} ({sim:.2f})")

    if now - staff_last_seen.get(key, 0) > CONFIG['COOLDOWN_MINUTES'] * 60 and staff["id"] is not None:
        post_staff_to_django(staff["id"], sim)
    staff_last_seen[key] = now

def mark_stranger(t, mean_emb, staff_sim, now):
    global latest_detection, latest_decision_at, last_vid_ms
    latest_decision_at = now
    with cache_lock:
        best_sim, matched = best_stranger_match(mean_emb)
        if matched is not None and best_sim >= CONFIG['STRANGER_THRESHOLD']:
            vid = matched["id"]
            count_visit = now - stranger_last_seen.get(vid, 0) > CONFIG['COOLDOWN_MINUTES'] * 60
            if count_visit:
                matched["visits"] += 1
            is_new, visits = False, matched["visits"]
        else:
            # ID из миллисекунд, но уникальный: двое, зашедшие вместе, решаются в одну и ту же миллисекунду
            ms = max(int(time.time() * 1000), last_vid_ms + 1)
            while f"str_{ms}" in known_strangers:
                ms += 1
            last_vid_ms = ms
            vid = f"str_{ms}"
            # Сразу кладём в кэш, чтобы другие треки этого же человека его узнали
            known_strangers[vid] = {"id": vid, "gallery": mean_emb[None, :], "visits": 1}
            is_new, count_visit, visits = True, True, 1

    label = "New Visitor" if is_new else f"Visitor (Visits: {visits})"
    t.update(kind="stranger", label=label, stranger_id=vid, familiar=not is_new, decided_at=now,
             pending={"vid": vid, "is_new": is_new, "count_visit": count_visit,
                      "prev_seen": stranger_last_seen.get(vid, 0)})
    stranger_last_seen[vid] = now

    if is_new:
        latest_detection = {"name": "Незнакомец", "status": "stranger", "time": time.strftime("%H:%M:%S"),
                            "match": 0, "access": "Учет прохожего"}
    else:
        latest_detection = {"name": f"Незнакомец {vid}", "status": "familiar", "time": time.strftime("%H:%M:%S"),
                            "match": int(best_sim * 100), "access": f"Повторный визит ({visits})"}
    nearest = f"{matched['id']} ({best_sim:.2f})" if matched is not None else "нет"
    log_decision(t, f"{'НОВЫЙ' if is_new else 'ЗНАКОМЫЙ'} незнакомец {vid} | "
                    f"ближайший незнакомец: {nearest}, ближайший сотрудник: {staff_sim:.2f}")

def cancel_stranger(t):
    p, t["pending"] = t["pending"], None
    with cache_lock:
        s = known_strangers.get(p["vid"])
        if s is None:
            return
        if p["is_new"]:
            del known_strangers[p["vid"]]
        elif p["count_visit"]:
            s["visits"] -= 1
    stranger_last_seen[p["vid"]] = p["prev_seen"]

def commit_stranger(t):
    """Отправка в Django откладывается на пару секунд: за это время трек набирает больше кадров,
    и в базу уходит эталон по лучшим кадрам и лучшее фото, а не первые 3 мутных кадра."""
    p = t.get("pending")
    if p is None:
        return
    t["pending"] = None
    template = track_template(t)
    with cache_lock:
        s = known_strangers.get(p["vid"])
        if s is None:
            return
        if p["is_new"]:
            s["gallery"][0] = template  # черновой эталон из первых кадров -> итоговый по всему треку
        else:
            add_to_gallery(s, template)
        db_emb = gallery_centroid(s)
    if p["is_new"] or p["count_visit"]:
        post_to_django(p["vid"], db_emb, t["best_crop"] if p["is_new"] else None)

def sharp_evidence(t):
    """Только чёткие кадры доказывают, что это НЕ сотрудник: на смазанных сходство падает и у своих.
    Возвращает (число чётких кадров, их суммарное качество, их среднее лицо)."""
    idx = [i for i, b in enumerate(t["blurs"]) if b >= CONFIG['STRANGER_MIN_BLUR']]
    if not idx:
        return 0, 0.0, None
    w = np.asarray([t["weights"][i] for i in idx], dtype=np.float32)[:, None]
    emb = (np.asarray([t["embs"][i] for i in idx]) * w).sum(axis=0)
    return len(idx), float(sum(t["qs"][i] for i in idx)), emb / np.linalg.norm(emb)

def finalize_on_exit(t, now):
    """Трек истёк без решения — человек быстро прошёл. Раньше такой трек молча выбрасывался."""
    if t["kind"] is not None or not t["embs"]:
        return
    n_sharp, mass, sharp_emb = sharp_evidence(t)
    if n_sharp < CONFIG['EXIT_MIN_FRAMES'] or mass < CONFIG['EXIT_MIN_QUALITY']:
        return log_decision(t, f"ушёл без решения: мало чётких кадров ({n_sharp})")
    sharp_sim, staff = best_staff_match(sharp_emb)
    frame_max = max(best_staff_match(e)[0] for e in t["embs"])
    # Пороги сотрудника на выходе не ослабляем: похожего на сотрудника лучше не записать никем, чем записать чужим
    if sharp_sim >= CONFIG['STAFF_GRAY_ZONE'] or frame_max >= CONFIG['STAFF_THRESHOLD']:
        return log_decision(t, f"ушёл без решения: похож на {staff['folder']} ({sharp_sim:.2f}/{frame_max:.2f})")
    mark_stranger(t, sharp_emb, sharp_sim, now)

def update_identity(t, frame_emb, now):
    n = t["samples"]
    mean_emb = track_template(t)
    staff_sim, staff = best_staff_match(mean_emb)
    frame_sim, frame_staff = best_staff_match(frame_emb)

    # === СОТРУДНИК === (проверяется и после решения "незнакомец": первые кадры могли быть плохими)
    if n >= CONFIG['CONFIRM_FRAMES'] and staff_sim >= CONFIG['STAFF_THRESHOLD']:
        return mark_staff(t, staff, staff_sim, now)
    if frame_sim >= CONFIG['FAST_ACCEPT']:
        return mark_staff(t, frame_staff, frame_sim, now)

    # === НЕЗНАКОМЕЦ === решаем только когда набралось достаточно чётких кадров
    if t["kind"] is not None:
        return
    n_sharp, mass, sharp_emb = sharp_evidence(t)
    if n_sharp < CONFIG['MIN_STRANGER_FRAMES'] or mass < CONFIG['MIN_STRANGER_QUALITY']:
        return
    sharp_sim, _ = best_staff_match(sharp_emb)
    if sharp_sim >= CONFIG['STAFF_GRAY_ZONE'] and n_sharp < CONFIG['MAX_DECISION_FRAMES']:
        return
    mark_stranger(t, sharp_emb, sharp_sim, now)

# ==========================================================
# 6. ТРЕКИНГ
# ==========================================================
camera = None
tracks = {}
tracks_lock = threading.Lock()

def ceil32(x):
    return int(((x + 31) // 32) * 32)

def get_roi_rect(frame_shape):
    h, w = frame_shape[:2]
    fx1, fy1, fx2, fy2 = CONFIG['ROI']
    return int(w * fx1), int(h * fy1), int(w * fx2), int(h * fy2)

def new_track(tid, box, now):
    return {"id": tid, "box": box, "time": now, "frame_time": now, "vel": (0.0, 0.0),
            "embs": [], "weights": [], "qs": [], "blurs": [], "samples": 0,
            "kind": None, "label": None, "familiar": False, "best_q": 0.0, "best_crop": None, "pending": None}

def update_track_box(t, box, frame_time, now):
    """Запоминаем скорость движения лица, чтобы при отрисовке рамка не отставала от свежего кадра."""
    dt = frame_time - t["frame_time"]
    if 0 < dt < 1.0:
        (ox1, oy1, ox2, oy2), (nx1, ny1, nx2, ny2) = t["box"], box
        vx = ((nx1 + nx2) - (ox1 + ox2)) / 2 / dt
        vy = ((ny1 + ny2) - (oy1 + oy2)) / 2 / dt
        t["vel"] = (0.5 * t["vel"][0] + 0.5 * vx, 0.5 * t["vel"][1] + 0.5 * vy)
    t["box"], t["time"], t["frame_time"] = box, now, frame_time

def needs_embedding(t):
    # Сотруднику эмбеддинги больше не нужны; незнакомцу — пока копим кадры перед отправкой в базу
    return t["kind"] is None or t["pending"] is not None

def add_sample(t, sample):
    t["embs"].append(sample["emb"])
    t["weights"].append(sample["w"])  # вес в среднем лице: качество с учётом смаза
    t["qs"].append(sample["q"])
    t["blurs"].append(sample["blur"])
    t["samples"] += 1
    if sample["w"] > t["best_q"]:
        t["best_q"], t["best_crop"] = sample["w"], sample["crop"]
    if len(t["embs"]) > CONFIG['TRACK_MAX_EMBS']:
        worst = int(np.argmin(t["weights"]))
        for key in ("embs", "weights", "qs", "blurs"):
            del t[key][worst]

def predicted_box(t, at_time, max_lag):
    """Где рамка трека окажется к моменту at_time, если человек продолжит двигаться так же."""
    x1, y1, x2, y2 = t["box"]
    lag = min(max(0.0, at_time - t["frame_time"]), max_lag)
    w = x2 - x1
    dx = int(np.clip(t["vel"][0] * lag, -2 * w, 2 * w))
    dy = int(np.clip(t["vel"][1] * lag, -2 * w, 2 * w))
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy

def match_detections(dets, tracks, frame_time):
    """Сопоставление лиц с треками один-к-одному. Раньше два лица рядом могли попасть в один трек,
    и эмбеддинги двух разных людей смешивались."""
    predicted = {tid: predicted_box(t, frame_time, CONFIG['HOLD_SECONDS']) for tid, t in tracks.items()}
    pairs = []
    for di, det in enumerate(dets):
        for tid, box in predicted.items():
            iou = compute_iou(det["box"], box)
            if iou > CONFIG['MIN_IOU']:
                pairs.append((iou, di, tid))
    assignment, used = {}, set()
    for _, di, tid in sorted(pairs, reverse=True):
        if di in assignment or tid in used:
            continue
        assignment[di] = tid
        used.add(tid)

    # Быстрое движение при медленном анализе: рамки уже не пересекаются, но центр рядом
    for di, det in enumerate(dets):
        if di in assignment:
            continue
        best_tid, best_d = None, 0.6
        for tid, box in predicted.items():
            if tid in used:
                continue
            d = center_distance(det["box"], box)
            if d < best_d:
                best_tid, best_d = tid, d
        if best_tid is not None:
            assignment[di] = best_tid
            used.add(best_tid)
    return assignment

def match_by_appearance(dets, tracks, assignment):
    """Человек, идущий поперёк кадра, между двумя анализами может сместиться больше чем на ширину лица —
    рамки не пересекаются, и раньше заводился новый трек (а решение "незнакомец" начиналось заново).
    Такие лица узнаём по эмбеддингу среди треков, оставшихся без пары."""
    used = set(assignment.values())
    templates = {tid: track_template(t) for tid, t in tracks.items() if tid not in used and t["embs"]}
    pairs = []
    for di, det in enumerate(dets):
        if di in assignment or det.get("sample") is None:
            continue
        for tid, template in templates.items():
            sim = float(np.dot(det["sample"]["emb"], template))
            if sim >= 0.5:
                pairs.append((sim, di, tid))
    for _, di, tid in sorted(pairs, reverse=True):
        if di in assignment or tid in used:
            continue
        assignment[di] = tid
        used.add(tid)

def analysis_loop():
    global latest_detection, analysis_period
    next_id = 0
    last_id = -1
    prepared_for = None
    det_scale = 1.0
    live_detector, det_size = None, None
    perf = {"since": time.time(), "cycles": 0, "faces": 0, "embeds": 0, "frame_id": camera.frame_id}

    while True:
        try:
            if not camera.is_connected or camera.frame_id == last_id:
                time.sleep(0.01)
                continue

            started = time.time()
            last_id = camera.frame_id
            frame = camera.frame

            rx1, ry1, rx2, ry2 = get_roi_rect(frame.shape)
            if prepared_for != (rx1, ry1, rx2, ry2):
                roi_w, roi_h = rx2 - rx1, ry2 - ry1
                det_w = CONFIG['DET_MAX_WIDTH']
                if det_w == 'auto':
                    det_w = max(640, ceil32(roi_w / 3))  # лицо в 50 px остаётся ≥16 px во входе детектора
                det_scale = min(1.0, det_w / roi_w)
                det_size = (ceil32(roi_w * det_scale), ceil32(roi_h * det_scale))
                live_detector = make_fixed_detector(det_size)
                prepared_for = (rx1, ry1, rx2, ry2)

            roi_img = np.ascontiguousarray(frame[ry1:ry2, rx1:rx2])
            bboxes, kpss = live_detector.detect(roi_img, input_size=det_size, max_num=0, metric='default')

            dets = []
            if bboxes is not None:
                for i in range(len(bboxes)):
                    score = float(bboxes[i, 4])
                    if score < CONFIG['DET_SCORE_MIN']:
                        continue
                    x1, y1, x2, y2 = (bboxes[i, :4] + np.array([rx1, ry1, rx1, ry1])).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if (x2 - x1) < CONFIG['MIN_FACE_SIZE'] or (y2 - y1) < CONFIG['MIN_FACE_SIZE']:
                        continue
                    kps = kpss[i] + np.array([rx1, ry1]) if kpss is not None else None
                    dets.append({"box": (x1, y1, x2, y2), "score": score, "kps": kps})

            # Треки меняет только этот поток, поэтому читать их можно без блокировки.
            # Тяжёлая часть (эмбеддинги) идёт вне блокировки — видео не подвисает на время распознавания.
            assignment = match_detections(dets, tracks, started)
            todo = []
            for di, det in enumerate(dets):
                t = tracks.get(assignment.get(di))
                det["reverify"] = bool(t is not None and t["embs"] and
                                       started - t["frame_time"] > max(CONFIG['REVERIFY_GAP'], 2.5 * analysis_period))
                if t is None or det["reverify"] or needs_embedding(t):
                    # 0 — проверка "тот ли это человек" (всегда), остальные — по числу уже набранных кадров:
                    # новые лица первыми, затем незнакомцы в окне проверки (может оказаться сотрудником),
                    # а давно висящие нераспознанные лица — в конце, чтобы не отнимать очередь у всех остальных.
                    prio = 0 if det["reverify"] else 1
                    todo.append((prio, t["samples"] if t is not None else 0, -(det["box"][2] - det["box"][0]), di))
            # Лимит распознаваний на кадр: иначе при толпе цикл растягивается, и новый человек ждёт своей очереди
            batch = []
            for prio, _, _, di in sorted(todo):
                if prio > 0 and len(batch) >= EMBEDS_PER_FRAME:
                    break
                sample = prepare_sample(frame, dets[di], det_scale)
                if sample is not None:
                    dets[di]["sample"] = sample
                    batch.append(sample)
            embed_samples(batch)
            for di, det in enumerate(dets):
                if det.get("sample") is not None and det["sample"]["emb"] is None:
                    det["sample"] = None
                # У двери люди идут друг за другом: следующий не должен унаследовать трек (и имя) предыдущего
                if (det["reverify"] and det.get("sample") is not None and
                        float(np.dot(det["sample"]["emb"], track_template(tracks[assignment[di]]))) < 0.3):
                    del assignment[di]
            match_by_appearance(dets, tracks, assignment)

            # Время ставим после обработки, а не до неё: раньше к моменту отрисовки трек уже
            # "устаревал" на время детекции+распознавания, и рамка не рисовалась вообще
            now = time.time()
            with tracks_lock:
                for di, det in enumerate(dets):
                    tid = assignment.get(di)
                    if tid is None:
                        tid = next_id
                        next_id += 1
                        tracks[tid] = new_track(tid, det["box"], now)
                    t = tracks[tid]
                    update_track_box(t, det["box"], started, now)

                    if t["kind"] == "staff":
                        staff_last_seen[t["staff_key"]] = now
                    elif t["kind"] == "stranger":
                        stranger_last_seen[t["stranger_id"]] = now

                    sample = det.get("sample")
                    if sample is not None and needs_embedding(t):
                        add_sample(t, sample)
                        update_identity(t, sample["emb"], now)

                    if t["pending"] is not None and (t["samples"] >= CONFIG['TRACK_MAX_EMBS'] or
                                                     now - t["decided_at"] >= CONFIG['STRANGER_COMMIT_SECONDS']):
                        commit_stranger(t)

                for k in [k for k, v in tracks.items() if now - v["time"] > CONFIG['HOLD_SECONDS']]:
                    t = tracks.pop(k)
                    finalize_on_exit(t, now)
                    commit_stranger(t)

                # Решение "на выходе" принимается, когда человека уже нет в кадре: даём панели его показать
                if not tracks and now - latest_decision_at > 5.0:
                    latest_detection = {"name": "-", "status": "waiting", "time": "-", "match": 0, "access": "-"}

            spent = time.time() - started
            analysis_period = 0.8 * analysis_period + 0.2 * spent

            perf["cycles"] += 1
            perf["faces"] += len(dets)
            perf["embeds"] += len(batch)
            elapsed = now - perf["since"]
            if CONFIG['PERF_LOG_SECONDS'] and elapsed >= CONFIG['PERF_LOG_SECONDS']:
                print(f"⏱ Анализ: {perf['cycles'] / elapsed:.1f} кадр/с ({analysis_period * 1000:.0f} мс на кадр), "
                      f"лиц в кадре {perf['faces'] / perf['cycles']:.1f}, распознаваний {perf['embeds'] / elapsed:.1f}/с, "
                      f"камера {(camera.frame_id - perf['frame_id']) / elapsed:.0f} fps, открыто панелей: {hub.clients}")
                perf.update(since=now, cycles=0, faces=0, embeds=0, frame_id=camera.frame_id)
            if spent < CONFIG['ANALYZE_INTERVAL']:
                time.sleep(CONFIG['ANALYZE_INTERVAL'] - spent)

        except Exception as e:
            print(f"⚠️ Ошибка в главном потоке: {e}")
            traceback.print_exc()
            time.sleep(1)

def render_jpeg(frame, connected):
    """Кадр для панели: сначала уменьшаем, потом рисуем рамки (так в разы дешевле, чем наоборот)."""
    h, w = frame.shape[:2]
    k = min(1.0, CONFIG['OUT_WIDTH'] / w)
    out = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_LINEAR) if k < 1.0 else frame.copy()
    if connected:
        if CONFIG['SHOW_ROI']:
            rx1, ry1, rx2, ry2 = get_roi_rect(frame.shape)
            cv2.rectangle(out, (int(rx1 * k), int(ry1 * k)), (int(rx2 * k), int(ry2 * k)), (255, 0, 255), 1)

        # Если анализ медленнее DRAW_TIMEOUT, фиксированный таймаут прятал все рамки
        draw_timeout = min(CONFIG['HOLD_SECONDS'], max(CONFIG['DRAW_TIMEOUT'], 2.5 * analysis_period))
        with tracks_lock:
            now = time.time()
            # Рамка посчитана по кадру, снятому какое-то время назад, — сдвигаем её туда, где лицо сейчас
            snapshot = [(predicted_box(t, now, 0.6), t["kind"], t["label"], t["familiar"])
                        for t in tracks.values() if now - t["time"] < draw_timeout]

        for (x1, y1, x2, y2), kind, label, familiar in snapshot:
            if kind == "staff":
                color, text = (0, 255, 0), label
            elif kind == "stranger":
                color, text = ((255, 200, 0) if familiar else (0, 165, 255)), label
            else:
                color, text = (160, 160, 160), "..."
            p1, p2 = (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k))
            cv2.rectangle(out, p1, p2, color, 2)
            cv2.putText(out, text, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None

class MjpegHub:
    """Один кодировщик видео на все открытые вкладки. Раньше каждая вкладка кодировала свой поток
    (~полядра CPU на вкладку), и всё это отнималось у распознавания. Без зрителей не кодирует вообще."""
    def __init__(self, fps):
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.clients, self.fps = None, 0, 0, fps
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last_id = -1
        while True:
            started = time.time()
            if self.clients == 0 or camera is None or camera.frame_id == last_id:
                time.sleep(0.01)
                continue
            last_id = camera.frame_id
            try:
                jpeg = render_jpeg(camera.frame, camera.is_connected)
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)
                continue
            if jpeg is not None:
                with self.cond:
                    self.jpeg, self.seq = jpeg, self.seq + 1
                    self.cond.notify_all()
            spare = 1.0 / self.fps - (time.time() - started)
            if spare > 0:
                time.sleep(spare)

    def frames(self):
        with self.cond:
            self.clients += 1
        seen = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq != seen, timeout=2.0)
                    jpeg, seen = self.jpeg, self.seq
                if jpeg is not None:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        finally:
            with self.cond:
                self.clients -= 1

hub = MjpegHub(CONFIG['STREAM_FPS'])

@flask_app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@flask_app.route('/video_feed')
def video_feed():
    return Response(hub.frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route('/status')
def get_status():
    return jsonify(latest_detection)

if __name__ == '__main__':
    camera = CameraStream(CONFIG['RTSP_URL'])
    threading.Thread(target=analysis_loop, daemon=True).start()
    threading.Thread(target=stranger_sync_loop, daemon=True).start()
    url = f"http://127.0.0.1:{CONFIG['PORT']}"
    print(f"🌐 Панель запущена: {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    flask_app.run(host='0.0.0.0', port=CONFIG['PORT'], debug=False, use_reloader=False, threaded=True)
