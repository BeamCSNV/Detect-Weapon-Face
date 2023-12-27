import cv2
from ultralytics import YOLO
import dlib
import requests
import numpy as np

LINE_NOTIFY_TOKEN = "qkG4N8B8Pk8W0nybCY1Plec1sibQ9kf9sSDYzrSk43s"

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

weapon_model = YOLO("best_lastest.pt")
face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        weapon_results = weapon_model.track(frame, conf=0.5)

        if weapon_results:
            # ตรวจสอบว่ามีใบหน้าหรือไม่
            faces = face_detector(frame)

            # กรณีตรวจเจออาวุธและใบหน้าพร้อมกัน
            if faces and len(weapon_results) == 1:
                # แคปภาพอาวุธและใบหน้า
                weapon_image_array = weapon_results[0].orig_img
                weapon_frame = weapon_results[0].plot()
                face_image = frame[faces[0].top():faces[0].top() + faces[0].height(),
                                faces[0].left():faces[0].left() + faces[0].width()]
                face_image_resized = cv2.resize(face_image, (weapon_image_array.shape[1], weapon_image_array.shape[0]),
                                               interpolation=cv2.INTER_AREA)
                weapon_and_face_image = np.concatenate((weapon_image_array, face_image_resized), axis=2)

                # วาดกรอบสี่เหลี่ยมที่ใบหน้า
                x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
                cv2.rectangle(weapon_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # บันทึกภาพอาวุธและใบหน้า
                cv2.imwrite("weapon_and_face.jpg", weapon_and_face_image)

                # ส่งข้อความแจ้งเตือนไปยังไลน์
                response = send_line_notify(
                    "พบผู้ร้ายพกอาวุธ", "weapon_and_face.jpg", "อาวุธและใบหน้า"
                )

    cv2.imshow("Detection Result", weapon_frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
