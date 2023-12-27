import face_recognition
import os
import pickle

# ระบุโฟลเดอร์ที่มีรูปภาพของแต่ละบุคคล
dataset_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/dataset"

# เก็บข้อมูลใบหน้าและชื่อ
known_encodings = []
known_names = []

# วนลูปผ่านรูปภาพในโฟลเดอร์ dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            image_path = os.path.join(person_folder, filename)

            # โหลดรูปภาพและหา face encoding
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)

            if len(face_encoding) > 0:
                # ให้ใบหน้าในรูปนี้มีชื่อ "Custom_Name" แทน
                known_encodings.append(face_encoding[0])
                known_names.append("Custom_Name")

# บันทึกข้อมูลใบหน้าในไฟล์ pickle
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("Encoding completed. 'encodings.pickle' file created.")
