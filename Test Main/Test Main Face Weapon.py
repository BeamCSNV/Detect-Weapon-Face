# นำเข้าไลบรารีที่จำเป็น
import os
import cv2
from ultralytics import YOLO
import dlib
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import time
import random
import requests

# กำหนดค่า TOKEN สำหรับ LINE Notify
LINE_NOTIFY_TOKEN = "6n6NUX9mdUjHysH3IqGEdyuN5Wmh6BPZwaZNlzuJr44"

# กำหนดค่าเริ่มต้นของตัวแปร currentname
currentname = "ไม่รู้จัก"

# กำหนดที่อยู่ของไฟล์ encodings.pickle
encodingsP = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Test Main/encodings.pickle"
print("[INFO] กำลังโหลด encodings + ตัวตรวจหน้า...")

# โหลดข้อมูล encodings จากไฟล์
data = pickle.loads(open(encodingsP, "rb").read())

# กำหนดที่อยู่ของโฟลเดอร์ dataset
dataset_path = "C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# โหลดโมเดล YOLO
model = YOLO('best_lastest.pt')

# ใช้ dlib เพื่อตรวจหน้าใบหน้า
face_detector = dlib.get_frontal_face_detector()

# เริ่มต้นการใช้งานกล้อง
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)

# เริ่มต้นการวัด Frames Per Second (FPS)
fps = FPS().start()

# ฟังก์ชันสำหรับส่งข้อความแจ้งเตือนผ่าน LINE Notify
def send_line_notify(message, image_path=None):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + LINE_NOTIFY_TOKEN}
    payload = {"message": message}
    files = {}

    if image_path:
        files = {"imageFile": open(image_path, "rb")}

    requests.post(url, headers=headers, params=payload, files=files)

# ลูปการทำงานตลอดเวลา
while True:
    # อ่าน Frame จากกล้อง
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # ตรวจหน้าใบหน้า
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []
    
    # ใช้ YOLO ในการตรวจจับวัตถุ
    results = model.track(frame, imgsz=640, stream=True, conf=0.5)

    # ตรวจสอบใบหน้าที่ตรงกับ encodings
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "ไม่รู้จัก"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(currentname)
        else:
            num = random.random()
            data["names"].append(str(num))
            data["encodings"].append(encoding)
            print(data["names"])

            # บันทึกภาพใบหน้าใหม่
            face_id = len(os.listdir(dataset_path)) + 1
            face_folder_path = os.path.join(dataset_path, f"ID{face_id:02d}")

            if not os.path.exists(face_folder_path):
                os.makedirs(face_folder_path)

            face_filename = os.path.join(face_folder_path, f"{face_id:02d}.jpg")
            cv2.imwrite(face_filename, frame[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][1]])

            # บันทึกข้อมูลใบหน้าใหม่ในไฟล์ encodings.pickle
            train_data = {"encodings": [encoding], "names": [str(num)]}
            with open(encodingsP, "wb") as f:
                f.write(pickle.dumps(train_data))

            # ส่งข้อความแจ้งเตือน LINE Notify
            send_line_notify("ตรวจพบใบหน้าใหม่!", image_path=face_filename)

        names.append(name)

    # วาดกรอบและข้อความบนภาพ
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        
    # วาดกรอบลงทับกับผลลัพธ์จาก YOLO
    for result in results:
        frame_ = result.plot()

    # แสดงภาพที่ได้ผลลัพธ์
    cv2.imshow("การระบุใบหน้ากำลังทำงาน", frame)

    # รอรับคีย์บอร์ด
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # อัปเดต FPS
    fps.update()

# หยุดการวัด FPS
fps.stop()
print("[INFO] เวลาที่ใช้: {:.2f}".format(fps.elapsed()))
print("[INFO] ประมาณ FPS: {:.2f}".format(fps.fps()))

# ปิดหน้าต่าง
cv2.destroyAllWindows()

# หยุดการใช้งานกล้อง
vs.stop()
