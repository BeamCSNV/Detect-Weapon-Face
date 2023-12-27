import cv2
import dlib
import requests
import numpy as np
from ultralytics import YOLO
import os
import pickle

LINE_NOTIFY_TOKEN = "6n6NUX9mdUjHysH3IqGEdyuN5Wmh6BPZwaZNlzuJr44"

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

# โมเดล YOLO สำหรับตรวจจับอาวุธ
weapon_model = YOLO("best_lastest.pt")

# ตัวตรวจจับใบหน้าจาก Dlib
face_detector = dlib.get_frontal_face_detector()

# เปิดการใช้งานกล้อง
cap = cv2.VideoCapture(0)

# ระบุ path ที่ต้องการเซฟรูปภาพ
dataset_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/dataset"

# ตรวจสอบว่าโฟลเดอร์ dataset ถูกสร้างขึ้นแล้วหรือยัง
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# ตรวจสอบว่าไฟล์ encodings.pickle มีอยู่หรือไม่
encodings_file_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/encodings.pickle"
if os.path.exists(encodings_file_path):
    with open(encodings_file_path, "rb") as encodings_file:
        encodings_data = pickle.load(encodings_file)
    known_face_encodings = encodings_data["encodings"]
    known_face_names = encodings_data["names"]
else:
    known_face_encodings = []
    known_face_names = []

while True:
    ret, frame = cap.read()

    if ret:
        # ใช้ YOLO เพื่อตรวจจับอาวุธ
        weapon_results = weapon_model.track(frame, conf=0.5)
        faces = face_detector(frame)

        # ตรวจสอบว่ามีใบหน้าหรือไม่
        if faces:
            # ตรวจสอบแต่ละใบหน้า
            for face in faces:
                # ตรวจสอบว่าใบหน้านี้เคยถูกแจ้งเตือนและเซฟรูปไว้แล้วหรือไม่
                face_image = frame[face.top():face.top() + face.height(), face.left():face.left() + face.width()]
                face_id = hash(face_image.tobytes())

                # ตรวจสอบว่า ID ของใบหน้านี้เคยได้รับการแจ้งเตือนหรือไม่
                if not os.path.exists(os.path.join(dataset_path, str(face_id))):
                    # ถ้าโฟลเดอร์ใหม่ถูกสร้างขึ้นเพื่อใบหน้านี้เท่านั้น
                    # บันทึกรูปใบหน้าใหม่
                    face_folder_path = os.path.join(dataset_path, str(face_id))
                    os.makedirs(face_folder_path)

                    face_filename = os.path.join(face_folder_path, f"{face_id}.jpg")
                    cv2.imwrite(face_filename, face_image)

                    # เพิ่มข้อมูลใบหน้าลงในรายชื่อที่จะใช้สร้างไฟล์ encodings.pickle
                    known_face_encodings.append(face_id)
                    known_face_names.append(str(face_id))

                    # ตั้งชื่อไฟล์สำหรับแจ้งเตือน
                    notification_filename = face_filename

                    # ส่งแจ้งเตือนไปยังไลน์พร้อมรูปภาพใบหน้าใหม่
                    response = send_line_notify("พบใบหน้าใหม่", notification_filename, "ใบหน้าใหม่")
                    print(response.text)

                    # บันทึกไฟล์ encodings.pickle ที่มีข้อมูลใบหน้าทั้งหมด
                    encodings_data = {"encodings": known_face_encodings, "names": known_face_names}
                    with open(encodings_file_path, "wb") as encodings_file:
                        pickle.dump(encodings_data, encodings_file)

        # ตรวจสอบว่ามีการตรวจจับอาวุธหรือไม่
        if faces and not weapon_results:
            continue
        elif faces and len(weapon_results) == 1:
            # ถ้าตรวจเจออาวุธและใบหน้าพร้อมกัน
            weapon_image_array = weapon_results[0].orig_img
            weapon_frame = weapon_results[0].plot()
            face_image = frame[faces[0].top():faces[0].top() + faces[0].height(),
                            faces[0].left():faces[0].left() + faces[0].width()]
            face_image_resized = cv2.resize(face_image, (weapon_frame.shape[1], weapon_frame.shape[0]),
                                            interpolation=cv2.INTER_AREA)
            weapon_and_face_image = np.concatenate((weapon_frame, face_image_resized), axis=1)

            # วาดกรอบตรวจจับใบหน้า
            x1_face, y1_face, x2_face, y2_face = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
            cv2.rectangle(weapon_and_face_image, (x1_face, y1_face), (x2_face, y2_face), (0, 0, 255), 2)

            # วาดกรอบตรวจจับอาวุธ
            x1_weapon, y1_weapon, x2_weapon, y2_weapon = 0, 0, weapon_frame.shape[1], weapon_frame.shape[0]
            cv2.rectangle(weapon_and_face_image, (x1_weapon, y1_weapon), (x2_weapon, y2_weapon), (0, 0, 255), 2)

            # ส่งข้อความแจ้งเตือนไปยังไลน์พร้อมรูปภาพและวิดีโอ
            # cv2.imwrite("weapon_and_face.jpg", weapon_and_face_image)
            # video_writer = cv2.VideoWriter("weapon_and_face.avi", cv2.VideoWriter_fourcc(*"XVID"), 10.0,
            # (weapon_and_face_image.shape[1], weapon_and_face_image.shape[0]))
            # video_writer.write(weapon_and_face_image)
            # video_writer.release()

            # response = send_line_notify("พบอาวุธและใบหน้าผู้ร้าย", "weapon_and_face.jpg", "อาวุธและใบหน้า")
            # print(response.text)

            # แสดงผลลัพธ์การตรวจจับ
            # cv2.imshow("Detection Result", weapon_and_face_image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

# ปิดการใช้งานกล้องและปิดหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
