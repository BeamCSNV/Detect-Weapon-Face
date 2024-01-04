import os
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import time
import random
import requests

LINE_NOTIFY_TOKEN = "6n6NUX9mdUjHysH3IqGEdyuN5Wmh6BPZwaZNlzuJr44"

currentname = "ไม่รู้จัก"

# โหลด face encodings และตัวตรวจหน้า
encodingsP = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Test Main/encodings.pickle"
print("[INFO] กำลังโหลด encodings + ตัวตรวจหน้า...")
data = pickle.loads(open(encodingsP, "rb").read())

# กำหนดที่เก็บ dataset
dataset_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)  # สร้างโฟลเดอร์เก็บรูปใบหน้าใหม่

# เริ่มต้น video stream และรอให้มันอุ่นขึ้น
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)

# เริ่มต้น FPS counter
fps = FPS().start()

# ฟังก์ชันสำหรับส่งข้อมูลไปยัง Line Notify
def send_line_notify(message, image_path=None):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + LINE_NOTIFY_TOKEN}

    payload = {"message": message}
    files = {}

    if image_path:
        files = {"imageFile": open(image_path, "rb")}

    requests.post(url, headers=headers, params=payload, files=files)

while True:
    # อ่านเฟรมจาก video stream และปรับขนาด
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # ตรวจหาตำแหน่งและ encodings ของใบหน้า
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        # เปรียบเทียบ face encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "ไม่รู้จัก"

        if True in matches:
            # ระบุบุคคลที่มีการตรงคู่มากที่สุด
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            # พิมพ์ชื่อถ้าต่างจากชื่อปัจจุบัน
            if currentname != name:
                currentname = name
                print(currentname)
        else:
            # เซฟรูปใบหน้าใน dataset และทำการเทรนใหม่เฉพาะใบหน้านี้
            num = random.random()
            data["names"].append(str(num))
            data["encodings"].append(encoding)
            print(data["names"])

            face_id = len(os.listdir(dataset_path)) + 1
            face_folder_path = os.path.join(dataset_path, f"ID{face_id:02d}")

            if not os.path.exists(face_folder_path):
                os.makedirs(face_folder_path)

            face_filename = os.path.join(face_folder_path, f"{face_id:02d}.jpg")
            cv2.imwrite(face_filename, frame[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][1]])

            # ทำการเทรนข้อมูลใหม่เฉพาะใบหน้านี้
            train_data = {"encodings": [encoding], "names": [str(num)]}
            with open(encodingsP, "wb") as f:
                f.write(pickle.dumps(train_data))

            # ส่งแจ้งเตือนไปยัง Line Notify เมื่อตรวจพบใบหน้าใหม่
            send_line_notify("ตรวจพบใบหน้าใหม่!", image_path=face_filename)

        names.append(name)

    # วาดสี่เหลี่ยมและชื่อลงบนเฟรม
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    # แสดงเฟรม
    cv2.imshow("การระบุใบหน้ากำลังทำงาน", frame)

    # ตรวจสอบปุ่ม 'q' เพื่อออกจากลูป
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # อัปเดต FPS counter
    fps.update()

# หยุด video stream และ FPS counter
fps.stop()
print("[INFO] เวลาที่ใช้: {:.2f}".format(fps.elapsed()))
print("[INFO] ประมาณ FPS: {:.2f}".format(fps.fps()))

# ปิดหน้าต่างทั้งหมด
cv2.destroyAllWindows()
vs.stop()
