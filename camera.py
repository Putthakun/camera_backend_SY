import fastface as ff
from keras_facenet import FaceNet
import cv2
import numpy as np

# Load FastFace model
model = ff.FaceDetector.from_pretrained("lffd_slim")

# Initialize the FaceNet embedder
embedder = FaceNet()

# เปิดกล้องหรืออ่านภาพจากไฟล์
cap = cv2.VideoCapture(0)  # เปิดกล้อง webcam (หรือใส่ path ของไฟล์วิดีโอ)

# กำหนดความละเอียด
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # ตั้งความกว้าง
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # ตั้งความสูง

while True:
    ret, frame = cap.read()  # อ่านภาพจากกล้อง
    if not ret:
        break

    # กลับภาพในแกน X (mirror)
    frame = cv2.flip(frame, 1)  # 1 คือการกลับในแกน X

    # Detect faces in the frame
    detections = model.predict(frame)

    # วาดกรอบรอบใบหน้าที่ตรวจพบ
    for detection in detections:
        boxes = detection['boxes']  # ดึง boxes จากการตรวจจับ
        scores = detection['scores']  # ดึงคะแนนความมั่นใจ

        for box, score in zip(boxes, scores):  # ใช้ zip เพื่อรวม boxes กับ scores
            if score >= 0.5:  # คะแนนความมั่นใจ
                x1, y1, x2, y2 = box  # ดึงพิกัดจาก box

                # ครอปใบหน้าจากเฟรม
                face_image = frame[y1:y2, x1:x2]

                # ตรวจสอบว่า face_image ไม่ว่างและมีขนาดที่ถูกต้อง
                if face_image.size > 0:
                    # เพิ่มมิติให้กับ face_image เพื่อให้ input เป็น 4 มิติ
                    face_image = cv2.resize(face_image, (160, 160))  # Resize ให้ตรงตามที่ FaceNet คาดหวัง
                    face_image = np.expand_dims(face_image, axis=0)  # เพิ่ม batch dimension

                    # ทำการสร้าง embedding
                    face_vector = embedder.embeddings(face_image)
                    print(face_vector)

                # วาดกรอบรอบใบหน้า
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # แสดงภาพ
    cv2.imshow("FastFace Detection", frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
