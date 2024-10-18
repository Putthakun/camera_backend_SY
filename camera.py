import cv2
import mediapipe as mp

# เริ่มต้น MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ไม่สามารถเปิดกล้องได้")
            break

        # แปลงภาพเป็น RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # ตรวจจับใบหน้า
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(image, bbox, (255, 0, 0), 2)
                mp_drawing.draw_detection(image, detection)

        # กลับภาพในแกน x (mirror)
        image = cv2.flip(image, 1)

        # แสดงภาพ
        cv2.imshow('Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # กด Esc เพื่อออก
            break

cap.release()
cv2.destroyAllWindows()
