import fastface as ff
import cv2

# Load FastFace model
model = ff.FaceDetector.from_pretrained("lffd_slim")

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # แสดงคะแนนความมั่นใจ
                text = f"{score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # แสดงภาพ
    cv2.imshow("FastFace Detection", frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
