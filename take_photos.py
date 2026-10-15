import cv2
import os


save_folder = 'known_faces/Anas'
os.makedirs(save_folder, exist_ok=True)

print("Подключение к камере...")
rtsp_url = "rtsp://admin:Danilaziz1021@192.168.88.68:554/stream2"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✅ Камера работает!")
print("Инструкция:")
print(" - Нажми 's', чтобы сделать фото")
print(" - Нажми 'q', чтобы выйти")

photo_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Photo Booth', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Если нажали 's' - сохраняем кадр
    if key == ord('s'):
        photo_count += 1
        filepath = os.path.join(save_folder, f"photo_{photo_count}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"📸 Фото {photo_count} сохранено в {filepath}!")

    # Если нажали 'q' - выходим
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()