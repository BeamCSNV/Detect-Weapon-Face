import cv2
from ultralytics import YOLO
import dlib
import requests

# ตัวแปร
LINE_NOTIFY_TOKEN = "qkG4N8B8Pk8W0nybCY1Plec1sibQ9kf9sSDYzrSk43s"

# ฟังก์ชั่น
def send_line_notify(message, image_path=None):
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

# โหลดโมเดล
weapon_model = YOLO('best_lastest.pt')
face_detector = dlib.get_frontal_face_detector()

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # อ่านเฟรมจากวิดีโอ

    if ret:
        # ตรวจจับอาวุธในเฟรม
        weapon_results = weapon_model.predict(frame, conf=0.5)

        if weapon_results:
            # ตรวจเจออาวุธ
            weapon_frame = weapon_results[0].plot()

            # วาดกรอบรอบใบหน้าที่ตรวจพบ
            faces = face_detector(frame, 1)
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(weapon_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # แคปปืน
            weapon_image_path = None

            # ส่งรูปปืนไปไลน์
            weapon_image_path = weapon_results[0].imgs[0]
            cv2.imshow('Detection Result', weapon_frame)  # แสดงภาพในหน้าต่าง
            
            # ส่งรูปปืนไปไลน์
            response = send_line_notify("ตรวจพบอาวุธ", weapon_image_path)

            # ตรวจจับใบหน้าในเฟรม
            faces = face_detector(frame)

            # แสดงแจ้งเตือน
            if response.status_code == 200:
                cv2.putText(weapon_frame, "Success", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                cv2.putText(weapon_frame, "Non Success", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            # ไม่มีอาวุธ
            pass

cap.release()
cv2.destroyAllWindows()
