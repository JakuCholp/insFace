import os
import requests

# Наша главная папка с базой
base_dir = 'known_faces'
os.makedirs(base_dir, exist_ok=True)

print("🚀 Запрашиваем 100 случайных личностей через API...")

# nat=us,gb гарантирует, что в именах будут только латинские буквы A-Z 
# (без турецких или французских символов с точками, от которых упадет OpenCV)
url = "https://randomuser.me/api/?results=100&nat=us,gb&inc=name,picture"
response = requests.get(url).json()

people = response['results']

for i, person in enumerate(people, 1):
    first_name = person['name']['first']
    last_name = person['name']['last']
    folder_name = f"{first_name}_{last_name}"
    
    # 1. Создаем папку под человека: known_faces/John_Smith/
    person_dir = os.path.join(base_dir, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # 2. Скачиваем его лицо
    img_url = person['picture']['large']
    img_data = requests.get(img_url).content
    
    # 3. Сохраняем в папку как face.jpg
    file_path = os.path.join(person_dir, "face.jpg")
    with open(file_path, 'wb') as f:
        f.write(img_data)
        
    print(f"[{i}/100] Добавлен в базу: {folder_name}")

print("\n🎉 Готово! Загляни в папку known_faces — там теперь живет 100 новых человек.")