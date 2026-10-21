"""Номера машин с той же камеры, что и лица.

Работает в своём потоке внутри main.py и берёт уже готовые кадры камеры: второго подключения к камере нет.
Номер ищется только в зоне PLATE_ROI и только когда там что-то движется, PLATE_FPS раз в секунду, чтобы не
отнимать процессор у лиц. Одна машина - один трек: номер читается на каждом кадре, а в Django уходит итог
голосования по всем кадрам, когда машина уехала из кадра или остановилась.

Ставить ничего не нужно: модели запускаются на том же onnxruntime, что и лица. При первом запуске они
(~13 МБ) скачиваются в папку plate_models/ рядом с этим файлом. Модели - из открытых проектов
open-image-models (поиск номера) и fast-plate-ocr (чтение) автора ankandrew; их библиотеки требуют
Python 3.10+, поэтому подготовка картинок и разбор ответа моделей повторены здесь.
"""
import os
import re
import time
import threading
import traceback
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import onnxruntime as ort
import requests

from plate_text import normalize_plate, plate_candidates, plate_format

# Если номер отличается от номера сотрудника одной такой парой символов - это ошибка чтения, а не другая машина
CONFUSABLE = [set(group) for group in ('0ODQ', '1IL7T', '2Z', '5S', '6G', '8B', 'MNH', 'KX', 'PR', 'UVY', 'CG', 'EF')]
DIRECTION_NAMES = {'entry': 'въезд', 'exit': 'выезд', 'unknown': 'направление неизвестно'}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plate_models')
DETECTOR_URL = 'https://github.com/ankandrew/open-image-models/releases/download/assets/'
OCR_URL = 'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/'


def download(url, name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"   скачиваю {name}...")
        res = requests.get(url + name, timeout=60)
        res.raise_for_status()
        with open(path + '.part', 'wb') as f:
            f.write(res.content)
        os.replace(path + '.part', path)  # оборванная загрузка не останется "готовой" моделью
    return path


def session(path, threads):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')  # не крутить потоки вхолостую
    so.intra_op_num_threads = threads
    # Только CPU: на CoreML у детектора не работает встроенный в модель отбор рамок (NMS).
    # Модели номеров маленькие, процессору они не в тягость.
    return ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])


class PlateDetector:
    """YOLOv9 с отбором рамок внутри модели: кадр -> [((x1, y1, x2, y2), уверенность)]."""

    def __init__(self, name, threads, min_score):
        self.sess = session(download(DETECTOR_URL, name), threads)
        self.input = self.sess.get_inputs()[0]
        self.size = tuple(self.input.shape[2:4])  # (высота, ширина) входа, например 384x384
        self.min_score = min_score

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.size[0] / h, self.size[1] / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.size[1] - nw) / 2, (self.size[0] - nh) / 2
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
        img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)),
                                 int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = (img.transpose(2, 0, 1)[::-1] / 255.0).astype(np.float32)[None]  # BGR -> RGB, NCHW
        out = np.asarray(self.sess.run(None, {self.input.name: x})[0])
        rows = out[0] if out.ndim == 3 else out[out[:, 0] == 0]  # [батч, x1, y1, x2, y2, класс, уверенность]
        boxes = []
        for row in rows:
            if row[6] < self.min_score:
                continue
            x1, y1, x2, y2 = (row[1] - dw) / r, (row[2] - dh) / r, (row[3] - dw) / r, (row[4] - dh) / r
            boxes.append(((int(x1), int(y1), int(x2), int(y2)), float(row[6])))
        return boxes


class PlateReader:
    """Чтение номера по вырезке: [картинки BGR] -> [(текст, уверенность по каждому символу)]."""

    def __init__(self, name, threads):
        path = download(OCR_URL, name)
        config = open(download(OCR_URL, name.replace('.onnx', '_plate_config.yaml')), encoding='utf-8').read()

        def value(key):
            m = re.search(r'^' + key + r":\s*'?([^'\n]*)'?\s*$", config, re.M)
            if m is None:
                raise ValueError(f"в настройках модели номеров нет {key}")
            return m.group(1).strip()
        self.alphabet, self.pad = value('alphabet'), value('pad_char')
        self.slots, self.height, self.width = int(value('max_plate_slots')), int(value('img_height')), int(value('img_width'))
        self.rgb = value('image_color_mode') == 'rgb'
        if value('keep_aspect_ratio').lower() == 'true':
            raise ValueError("модели с keep_aspect_ratio не поддерживаются")
        self.sess = session(path, threads)
        names = [o.name for o in self.sess.get_outputs()]
        self.output = 'plate' if 'plate' in names else names[0]

    def run(self, crops):
        batch = []
        for crop in crops:
            img = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if self.rgb else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            batch.append(img)
        x = np.stack(batch).astype(np.uint8)
        probs = self.sess.run([self.output], {self.sess.get_inputs()[0].name: x})[0]
        probs = probs.reshape((-1, self.slots, len(self.alphabet)))
        results = []
        for p in probs:
            text = ''.join(self.alphabet[i] for i in p.argmax(axis=-1)).rstrip(self.pad)
            results.append((text.replace(self.pad, ''), p.max(axis=-1)))
        return results


def roi_rect(shape, roi):
    h, w = shape[:2]
    return int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)


def format_plate(plate):
    """'01123ABC' -> '01 123 ABC': как написано на самом номере."""
    groups = {'KG': (2, 3), 'KG_ORG': (2, 3), 'KG_OLD': (1, 4), 'KZ': (3, -2), 'KZ_OLD': (1, 3), 'RU': (1, 3, 2)}
    cuts = groups.get(plate_format(plate))
    if cuts is None:
        return plate
    parts, rest = [], plate
    for n in cuts:
        if n < 0:
            parts += [rest[:n], rest[n:]]
            rest = ''
            break
        parts.append(rest[:n])
        rest = rest[n:]
    return ' '.join(p for p in parts + [rest] if p)


def vote(reads):
    """Итог по всем кадрам машины. reads: [(номер, вес)]. Возвращает (номер, согласие по каждому символу).

    Голосуем посимвольно среди прочтений самой частой длины: на одном кадре ошиблись в третьей букве,
    на другом в первой цифре - вместе они дают верный номер, хотя целиком он ни разу не прочитан."""
    by_len = defaultdict(float)
    for text, weight in reads:
        by_len[len(text)] += weight
    length = max(by_len, key=by_len.get)
    group = [(text, weight) for text, weight in reads if len(text) == length]
    chars, agree = [], []
    for i in range(length):
        votes = defaultdict(float)
        for text, weight in group:
            votes[text[i]] += weight
        char, best = max(votes.items(), key=lambda kv: kv[1])
        chars.append(char)
        agree.append(best / sum(votes.values()))
    return normalize_plate(''.join(chars)), agree


def snap_to_staff(plate, agree, staff_plates):
    """Номер сотрудника, если прочитанный от него отличается только ошибкой чтения."""
    for candidate in plate_candidates(plate):
        if candidate in staff_plates:
            return candidate
    # Прочитан кусок номера (часто без кода региона), который есть ровно у одного сотрудника
    if len(plate) >= 5 and not plate_format(plate):
        containing = [p for p in staff_plates if plate in p and len(p) > len(plate)]
        if len(containing) == 1:
            return containing[0]
    for staff_plate in staff_plates:
        if len(staff_plate) != len(plate):
            continue
        diff = [i for i in range(len(plate)) if plate[i] != staff_plate[i]]
        # Одна замена: похожие символы или символ, в котором кадры не сошлись. Две и больше - другая машина
        # (у служебных машин номера часто идут подряд: 01 123 ABC и 01 124 ABC)
        if len(diff) == 1:
            i = diff[0]
            if (i < len(agree) and agree[i] < 0.7) or any({plate[i], staff_plate[i]} <= g for g in CONFUSABLE):
                return staff_plate
    return plate


def movement(boxes):
    """(насколько вырос номер, насколько сдвинулся в ширинах номера) от начала трека к концу."""
    k = max(1, len(boxes) // 3)
    widths = [b[2] - b[0] for b in boxes]
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    w0, w1 = float(np.median(widths[:k])), float(np.median(widths[-k:]))
    c0, c1 = np.median(centers[:k], axis=0), np.median(centers[-k:], axis=0)
    return w1 / max(w0, 1.0), (c1 - c0) / max(float(np.median(widths)), 1.0)


def travel_direction(boxes, approach):
    """Номер растёт или опускается в кадре - машина едет к камере, уменьшается или поднимается - от неё."""
    growth, shift = movement(boxes)
    if growth > 1.15 or (growth >= 0.87 and shift[1] > 0.5):
        way = 'approach'
    elif growth < 0.87 or shift[1] < -0.5:
        way = 'away'
    else:
        return 'unknown'
    if way == 'approach':
        return approach
    return {'entry': 'exit', 'exit': 'entry'}.get(approach, 'unknown')


def has_moved(boxes):
    """Машина на стоянке тоже попадает в кадр, когда мимо идёт человек: такой номер не считается проездом."""
    growth, shift = movement(boxes)
    return abs(growth - 1.0) > 0.15 or float(np.hypot(*shift)) > 0.5


def car_photo(frame, box):
    """Номер вместе с передом машины (копия, а не ссылка на весь кадр): в журнале видно, что за машина."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    fh, fw = frame.shape[:2]
    return frame[max(0, y1 - 4 * h):min(fh, y2 + 2 * h), max(0, x1 - w):min(fw, x2 + w)].copy()


def encode_jpeg(img, quality=85):
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class PlateWatcher:
    def __init__(self, cfg, get_camera):
        self.cfg = cfg
        self.get_camera = get_camera
        print(f"🚗 Загрузка моделей номеров ({cfg['PLATE_DETECTOR']}, {cfg['PLATE_OCR']})...")
        self.detector = PlateDetector(cfg['PLATE_DETECTOR'], cfg['PLATE_THREADS'], cfg['PLATE_DET_SCORE'])
        self.ocr = PlateReader(cfg['PLATE_OCR'], cfg['PLATE_THREADS'])
        self.lock = threading.Lock()
        self.tracks = {}
        self.next_id = 0
        self.prev_small = None
        self.staff = {}               # номер сотрудника -> {"staff", "staff_name", "car_model"}
        self.journal = OrderedDict()  # номер -> карточка для панели; последние - в конце
        self.retry = []               # проезды, которые не дошли до Django
        self.boot = str(int(time.time()))
        self.perf = {"since": time.time(), "cycles": 0, "moving": 0, "det_ms": 0.0, "ocr_ms": 0.0, "reads": 0, "cars": 0}
        self.warned_secret = False
        threading.Thread(target=self.run, daemon=True).start()
        threading.Thread(target=self.sync_loop, daemon=True).start()
        print(f"✅ Номера машин: зона {cfg['PLATE_ROI']}, {cfg['PLATE_FPS']} раза в секунду при движении")

    # ---------------- анализ кадров ----------------
    def run(self):
        last_id = -1
        while True:
            started = time.time()
            try:
                camera = self.get_camera()
                if camera is None or not camera.is_connected or camera.frame_id == last_id:
                    time.sleep(0.05)
                    continue
                last_id = camera.frame_id
                self.step(camera.frame, started)
                self.log_perf()
            except Exception as e:
                print(f"⚠️ Номера: ошибка {e}")
                traceback.print_exc()
                time.sleep(1)
            rest = 1.0 / self.cfg['PLATE_FPS'] - (time.time() - started)
            if rest > 0:
                time.sleep(rest)

    def step(self, frame, now):
        rx1, ry1, rx2, ry2 = roi_rect(frame.shape, self.cfg['PLATE_ROI'])
        roi = frame[ry1:ry2, rx1:rx2]
        self.perf["cycles"] += 1
        if self.motion(roi):
            self.perf["moving"] += 1
            reads = self.read(frame, self.detect(roi, rx1, ry1))
            with self.lock:
                self.update_tracks(reads, now)
        with self.lock:
            done = [t for t in self.tracks.values() if now - t["last"] > self.cfg['PLATE_TRACK_GAP'] or
                    (not t["posted"] and now - t["first"] > self.cfg['PLATE_MAX_TRACK_SECONDS'])]
            for t in done:
                if now - t["last"] > self.cfg['PLATE_TRACK_GAP']:
                    del self.tracks[t["id"]]
        for t in done:
            if not t["posted"]:
                t["posted"] = True
                self.finish(t, now)

    def motion(self, roi):
        """Есть ли движение в зоне: сравниваем с прошлым проанализированным кадром в маленьком разрешении."""
        small = cv2.cvtColor(cv2.resize(roi, (160, max(1, int(160 * roi.shape[0] / roi.shape[1]))),
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        prev, self.prev_small = self.prev_small, small
        if prev is None or prev.shape != small.shape:
            return True
        return float(np.mean(cv2.absdiff(small, prev) > 18)) > self.cfg['PLATE_MOTION']

    def detect(self, roi, ox, oy):
        t0 = time.perf_counter()
        found = self.detector.predict(roi)
        self.perf["det_ms"] += (time.perf_counter() - t0) * 1000
        return [((x1 + ox, y1 + oy, x2 + ox, y2 + oy), score) for (x1, y1, x2, y2), score in found]

    def read(self, frame, boxes):
        """Читаем номера на вырезках из полного кадра (а не из уменьшенного входа детектора)."""
        h, w = frame.shape[:2]
        items, crops = [], []
        for (x1, y1, x2, y2), score in boxes:
            if x2 - x1 < self.cfg['PLATE_MIN_WIDTH']:
                continue
            # Модель чтения обучена на вырезках с полями: впритык она теряет крайние символы (код региона)
            px, py = int(0.15 * (x2 - x1)), int(0.35 * (y2 - y1))
            crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0:
                continue
            items.append(((x1, y1, x2, y2), score, crop))
            crops.append(crop)
        if not crops:
            return []
        t0 = time.perf_counter()
        predictions = self.ocr.run(crops)
        self.perf["ocr_ms"] += (time.perf_counter() - t0) * 1000
        reads = []
        for (box, score, crop), (raw, probs) in zip(items, predictions):
            conf = float(np.mean(probs[:len(raw)])) if len(raw) else 0.0
            plate = normalize_plate(raw)
            if len(plate) < 4:
                continue
            reads.append({"box": box, "raw": raw, "plate": plate, "conf": conf, "crop": crop.copy(),
                          "car": car_photo(frame, box), "width": box[2] - box[0]})
        self.perf["reads"] += len(reads)
        return reads

    def update_tracks(self, reads, now):
        """Одна машина - один трек: рядом с тем же номером на прошлых кадрах или с почти тем же текстом."""
        live = [t for t in self.tracks.values() if now - t["last"] <= self.cfg['PLATE_TRACK_GAP']]
        used = set()
        for read in sorted(reads, key=lambda r: -r["width"]):
            best, best_cost = None, None
            cx, cy = (read["box"][0] + read["box"][2]) / 2, (read["box"][1] + read["box"][3]) / 2
            for t in live:
                if t["id"] in used:
                    continue
                lb = t["boxes"][-1]
                dist = np.hypot(cx - (lb[0] + lb[2]) / 2, cy - (lb[1] + lb[3]) / 2) / max(read["width"], lb[2] - lb[0])
                same_text = sum(a != b for a, b in zip(read["plate"], t["guess"])) + abs(len(read["plate"]) - len(t["guess"]))
                # Машина у ворот за четверть секунды сдвигается на 1-2 ширины номера
                if dist > 4.0 and same_text > 2:
                    continue
                cost = dist + same_text
                if best_cost is None or cost < best_cost:
                    best, best_cost = t, cost
            if best is None:
                best = {"id": self.next_id, "first": now, "last": now, "boxes": [], "reads": [], "guess": read["plate"],
                        "agree": [], "posted": False, "label": read["plate"], "kind": None}
                self.tracks[best["id"]] = best
                self.next_id += 1
            used.add(best["id"])
            best["last"] = now
            best["boxes"].append(read["box"])
            best["reads"].append(read)
            if len(best["reads"]) > self.cfg['PLATE_MAX_READS']:
                # Храним самые крупные (самые читаемые) кадры, но первые и последние рамки для направления остаются
                best["reads"].remove(min(best["reads"][1:-1], key=lambda r: r["width"]))
            best["guess"], best["agree"] = vote([(r["plate"], self.weight(r)) for r in best["reads"]])
            best["label"] = snap_to_staff(best["guess"], best["agree"], self.staff)
            best["kind"] = "staff" if best["label"] in self.staff else None

    def weight(self, read):
        # Крупный и уверенно прочитанный номер весит больше: мелкий и смазанный чаще читается с ошибкой
        return max(read["conf"], 0.05) * min(1.0, read["width"] / 120.0)

    # ---------------- итог по машине ----------------
    def finish(self, t, now):
        reads = t["reads"]
        if len(reads) < self.cfg['PLATE_MIN_READS']:
            return
        if not has_moved(t["boxes"]):
            return  # стоит на месте - это не проезд
        plate = snap_to_staff(t["guess"], t["agree"], self.staff)
        direction = travel_direction(t["boxes"], self.cfg['PLATE_APPROACH'])
        best = max(reads, key=lambda r: (r["plate"] == t["guess"], self.weight(r)))
        confidence = round(min(t["agree"]) * float(np.mean([r["conf"] for r in reads])), 3)
        self.perf["cars"] += 1
        self.save_crops(plate, reads)
        photo = encode_jpeg(best["car"])
        card = self.log_card(plate, direction, best["crop"], now)
        print(f"🚗 Номер {format_plate(plate)}: {DIRECTION_NAMES[direction]}, кадров {len(reads)}, "
              f"согласие {min(t['agree']):.0%}, прочитано как {best['raw']}")
        item = {"data": {"plate": plate, "raw_text": best["raw"], "direction": direction,
                         "confidence": confidence, "camera": self.cfg['CAMERA_ID']},
                "photo": photo, "key": card["key"], "tries": 0}
        threading.Thread(target=self.send, args=(item,), daemon=True).start()

    def save_crops(self, plate, reads):
        """Вырезки номеров для проверки и будущего дообучения модели на своих номерах."""
        if not self.cfg['PLATE_SAVE_CROPS']:
            return
        folder = os.path.join('plate_crops', time.strftime('%Y-%m-%d'))
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime('%H%M%S')
        for i, read in enumerate(sorted(reads, key=lambda r: -r["width"])[:3]):
            cv2.imwrite(os.path.join(folder, f"{stamp}_{plate}_{i}_{read['raw'] or 'empty'}_{read['width']}px.jpg"),
                        read["crop"])

    # ---------------- Django ----------------
    def headers(self):
        key = self.cfg['CAMERA_SECRET_KEY']
        return {'X-Secret-Key': key} if key else {}

    def warn_secret(self, status):
        if status == 403 and not self.warned_secret:
            self.warned_secret = True
            print("⚠️ Django не принял ключ камеры. Задай одинаковый CAMERA_SECRET_KEY в env Django "
                  "и в переменной окружения перед запуском main.py")

    def send(self, item):
        try:
            files = {'photo': ('car.jpg', item["photo"], 'image/jpeg')} if item["photo"] else None
            res = requests.post(self.cfg['DJANGO_PLATES_URL'], data=item["data"], files=files,
                                headers=self.headers(), timeout=5)
        except requests.RequestException as e:
            self.queue_retry(item, str(e))
            return
        if res.status_code >= 500:
            self.queue_retry(item, f"{res.status_code}")
            return
        if res.status_code not in (200, 201):
            self.warn_secret(res.status_code)
            print(f"⚠️ Номер {item['data']['plate']} не принят Django: {res.status_code} {res.text[:300]}")
            return
        answer = res.json()
        self.update_card(item["key"], answer)
        who = answer.get('staff_name') if answer.get('is_staff') else 'гость'
        print(f"✅ Проезд {format_plate(answer.get('plate', item['data']['plate']))} записан: {who}, "
              f"визитов {answer.get('visits_count')}" + (" (повтор)" if answer.get('duplicate') else ""))

    def queue_retry(self, item, error):
        item["tries"] += 1
        with self.lock:
            if len(self.retry) < 200:
                self.retry.append(item)
        if item["tries"] == 1:
            print(f"⚠️ Django недоступен ({error}), проезд {item['data']['plate']} отправлю позже")

    def sync_loop(self):
        while True:
            try:
                res = requests.get(self.cfg['DJANGO_STAFF_PLATES_URL'], headers=self.headers(), timeout=5)
                if res.status_code == 200:
                    staff = {row['plate']: row for row in res.json()}
                    if staff.keys() != self.staff.keys():
                        print(f"🚗 Номеров сотрудников в базе: {len(staff)}")
                    self.staff = staff
                else:
                    self.warn_secret(res.status_code)
            except (requests.RequestException, ValueError):
                pass
            with self.lock:
                pending, self.retry = self.retry, []
            for item in pending:
                self.send(item)
            time.sleep(60 if pending else self.cfg['PLATE_SYNC_SECONDS'])

    # ---------------- панель ----------------
    def log_card(self, plate, direction, crop, now):
        thumb = None
        if crop is not None and crop.size:
            k = min(1.0, 200.0 / crop.shape[1])
            thumb = encode_jpeg(cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k)))))
        with self.lock:
            card = self.journal.pop(plate, None) or {"key": plate, "first_seen": now, "version": 0}
            card.update(plate=plate, display=format_plate(plate), direction=direction, last_seen=now,
                        kind="staff" if plate in self.staff else "pending",
                        name=self.staff.get(plate, {}).get('staff_name'), thumb=thumb,
                        version=card["version"] + 1)
            self.journal[plate] = card
            while len(self.journal) > self.cfg['PLATE_LOG_MAX']:
                self.journal.popitem(last=False)
            return card

    def update_card(self, key, answer):
        with self.lock:
            card = self.journal.get(key)
            if card is None:
                return
            card["kind"] = "staff" if answer.get('is_staff') else "guest"
            card["name"] = answer.get('staff_name') if answer.get('is_staff') else None
            card["visits"] = answer.get('visits_count')

    def cards(self):
        with self.lock:
            return [{**{k: v for k, v in c.items() if k != "thumb"}, "has_photo": c["thumb"] is not None,
                     "version": f"{self.boot}.{c['version']}"} for c in reversed(self.journal.values())]

    def thumb(self, key):
        with self.lock:
            card = self.journal.get(key)
            return card["thumb"] if card else None

    def draw(self, out, k, now):
        """Рамки номеров на видео панели: зелёная - сотрудник, оранжевая - гость, серая - номер ещё не прочитан."""
        with self.lock:
            shown = [(t["boxes"][-1], t["label"], t["kind"]) for t in self.tracks.values() if now - t["last"] < 1.0]
        for (x1, y1, x2, y2), label, kind in shown:
            color = (0, 255, 0) if kind == "staff" else (0, 165, 255) if len(label) >= 6 else (160, 160, 160)
            p1, p2 = (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k))
            cv2.rectangle(out, p1, p2, color, 2)
            # OpenCV не рисует кириллицу - имя сотрудника видно в журнале на панели
            text = format_plate(label) + (" staff" if kind == "staff" else "")
            cv2.putText(out, text, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def log_perf(self):
        p = self.perf
        elapsed = time.time() - p["since"]
        if not self.cfg['PERF_LOG_SECONDS'] or elapsed < self.cfg['PERF_LOG_SECONDS'] or not p["cycles"]:
            return
        moving = max(p["moving"], 1)
        print(f"🚗 Номера: {p['cycles'] / elapsed:.1f} кадр/с, движение в {p['moving'] / p['cycles']:.0%} кадров, "
              f"поиск номера {p['det_ms'] / moving:.0f} мс, чтение {p['ocr_ms'] / moving:.0f} мс, "
              f"прочтений {p['reads']}, машин {p['cars']}")
        self.perf = {"since": time.time(), "cycles": 0, "moving": 0, "det_ms": 0.0, "ocr_ms": 0.0, "reads": 0, "cars": 0}
