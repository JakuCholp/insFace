"""Снимки сотрудника С ТОЙ ЖЕ камеры, что и у main.py.

Фото с телефона и кадр с камеры наблюдения — очень разные картинки, и сходство между ними на видео
часто пограничное (0.35-0.45): человека узнаёт не с первого раза. Снимки с этой же камеры дают
сходство 0.6-0.9, и сотрудник узнаётся с первого кадра.

Запуск:   python take_photos.py 12_Anas        (12 — ID сотрудника в Django, Anas — имя)
Клавиши:  s — сохранить кадр, q — выйти.
Сделай 5-8 снимков там, где люди обычно проходят. Человек каждый раз останавливается на 1-2 секунды
лицом к камере (смазанные фото в базе вредят), на 2-3 разных расстояниях, чуть поворачивая голову.
Если у сотрудника уже есть папка (например, 'Anas'), переименуй её в '12_Anas' и снимай в неё же,
а не в новую. После этого перезапусти main.py.
"""
import os
import re
import sys
import time
import getpass
from urllib.parse import quote

import cv2

CAM_IP = os.environ.get("CAM_IP", "192.168.88.70")
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASSWORD = os.environ.get("CAM_PASSWORD") or getpass.getpass(f"Пароль камеры {CAM_IP}: ")
RTSP_URL = f"rtsp://{CAM_USER}:{quote(CAM_PASSWORD, safe='')}@{CAM_IP}:554/cam/realmonitor?channel=1&subtype=0"

folder = sys.argv[1] if len(sys.argv) > 1 else input("Папка сотрудника в формате ID_Имя (например 12_Anas): ").strip()
if not folder or os.sep in folder or folder.startswith('.'):
    sys.exit("❌ Нужно имя папки вида 12_Anas")
if not re.match(r'^\d+_', folder):
    print("⚠️ Имя папки без числового ID в начале — лог входа для этого сотрудника в Django не отправится.")
save_folder = os.path.join('known_faces', folder)
os.makedirs(save_folder, exist_ok=True)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
print(f"Подключение к камере {CAM_IP}...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    sys.exit("❌ Камера не открылась: проверь IP, логин и пароль")

print(f"✅ Камера работает. Снимки сохраняются в {save_folder}")
print("   s — сохранить кадр, q — выйти")

photo_count, failures = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        # Поток оборвался: не крутимся вхолостую, а переподключаемся
        failures += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if failures % 30 == 0:
            print("⚠️ Нет кадров с камеры, переподключаюсь...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        continue
    failures = 0

    h, w = frame.shape[:2]
    preview = cv2.resize(frame, (1280, int(h * 1280 / w))) if w > 1280 else frame
    cv2.imshow('Photo Booth', preview)

    key = cv2.waitKey(1) & 0xFF
    # Сохраняем полный кадр: main.py сам найдёт на нём самое крупное лицо
    if key == ord('s'):
        photo_count += 1
        filepath = os.path.join(save_folder, f"cam_{time.strftime('%Y%m%d_%H%M%S')}_{photo_count}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"📸 Фото {photo_count} сохранено: {filepath}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
