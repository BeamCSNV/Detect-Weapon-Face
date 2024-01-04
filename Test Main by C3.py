import cv2
import dlib
import requests
import numpy as np
import os
import pickle

# กำหนด Token สำหรับ Line Notify
LINE_NOTIFY_TOKEN = "6n6NUX9mdUjHysH3IqGEdyuN5Wmh6BPZwaZNlzuJr44"

# ฟังก์ชั่นสำหรับส่ง Line Notify
def send_line_notify(message, image_path=None, image_label=None):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + LINE_NOTIFY_TOKEN}
    payload = {"message": message}

    if image_path:
        image_data = open(image_path, "rb").read()
        files = {"imageFile": image_data}
        response = requests.post(url, headers=headers, data=payload, files=files)
    else:
        response = requests.post(url, headers=headers, data=payload)

    return response

# ตัวตรวจหน้าจาก dlib
face_detector = dlib.get_frontal_face_detector()

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนดที่เก็บ dataset
dataset_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# กำหนด path ของไฟล์ encodings
encodings_file_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/encodings.pickle"

# ถ้าไฟล์ encodings มีอยู่แล้วให้โหลดข้อมูล
if os.path.exists(encodings_file_path):
    with open(encodings_file_path, "rb") as encodings_file:
        encodings_data = pickle.load(encodings_file)
    known_face_encodings = encodings_data["encodings"]
    known_face_names = encodings_data["names"]
else:
    known_face_encodings = []
    known_face_names = []

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    if not ret:
        break

    # ตรวจสอบใบหน้าด้วย dlib
    faces = face_detector(frame)
    for face in faces:
        # ดึงข้อมูลใบหน้า
        face_image = frame[face.top():face.top() + face.height(), face.left():face.left() + face.width()]
        face_id = hash(face_image.tobytes())
        face_id_str = str(face_id)

        # ตรวจสอบว่าใบหน้านี้เคยเจอมาหรือไม่
        if face_id_str not in known_face_encodings:
            # วาดกรอบรอบใบหน้า
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # แสดงชื่อในภาพ
            cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # สร้างโฟลเดอร์เก็บรูปใบหน้าใหม่
            face_folder_path = os.path.join(dataset_path, face_id_str)
            if not os.path.exists(face_folder_path):
                os.makedirs(face_folder_path)

                # บันทึกรูปใบหน้า
                face_filename = os.path.join(face_folder_path, f"{face_id_str}.jpg")
                cv2.imwrite(face_filename, face_image)

                # เทรนใบหน้าใหม่
                known_face_encodings.append(face_id_str)
                known_face_names.append("Unknown")

                # แจ้งเตือน Line
                notification_filename = face_filename
                response = send_line_notify("พบใบหน้าใหม่", notification_filename, "ใบหน้าใหม่")
                print(response.text)

                # บันทึกข้อมูลใบหน้าใหม่ลงในไฟล์ encodings
                encodings_data = {"encodings": known_face_encodings, "names": known_face_names}
                with open(encodings_file_path, "wb") as encodings_file:
                    pickle.dump(encodings_data, encodings_file)

    # แสดงผลลัพธ์ทาง画面
    cv2.imshow("ผลลัพธ์การตรวจจับ", frame)

    # ตรวจสอบการกดปุ่ม q เพื่อปิดโปรแกรม
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# ปิดกล้อง
cap.release()

# ปิดหน้าต่าง
cv2.destroyAllWindows()
