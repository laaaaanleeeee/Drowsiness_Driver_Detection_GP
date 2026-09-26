import cv2
from ultralytics import YOLO
import time, threading
import pygame
import os

model = YOLO(r"D:\Graduation_Project\src\model\rs_ddd_2classes_yolov8n-20251207T210643Z-3-001\rs_ddd_2classes_yolov8n\weights\best.pt")

EYE_CLOSED_TIME_THRESHOLD = 0.8 
YAWN_TIME_THRESHOLD       = 1.0
ALERT_INTERVAL            = 3

pygame.mixer.init()
alert_sound = None
sound_path = r"D:\Graduation_Project\src\model\DuckQuack.mp3"

if os.path.exists(sound_path):
    try:
        alert_sound = pygame.mixer.Sound(sound_path)
    except:
        print("Không load được âm thanh!")
else:
    print("File âm thanh không tồn tại!")

def play_alert():
    try:
        alert_sound.play()
    except:
        pass

eye_time  = 0.0
yawn_time = 0.0
last_alert = 0.0

fps = 0
prev_time = 0

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # FPS
        now = time.time()
        delta_time = now - prev_time if prev_time else 0
        prev_time = now
        fps = fps * 0.9 + (1/delta_time)*0.1 if delta_time > 0 else fps

        results = model(frame, verbose=False)

        detected_eye = False
        detected_yawn = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:
                    detected_eye = True
                elif cls == 1: # yawning
                    detected_yawn = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        if detected_eye:
            eye_time += delta_time
        else:
            eye_time = max(0.0, eye_time - delta_time)

        if detected_yawn:
            yawn_time += delta_time
        else:
            yawn_time = max(0.0, yawn_time - delta_time)

        drowsy = False
        if eye_time >= EYE_CLOSED_TIME_THRESHOLD or yawn_time >= YAWN_TIME_THRESHOLD:
            drowsy = True
            if now - last_alert > ALERT_INTERVAL:
                threading.Thread(target=play_alert, daemon=True).start()
                last_alert = now

        cv2.putText(frame, f"EyeTime: {eye_time:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"YawnTime: {yawn_time:.2f}s",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        status_text = "DROWSY" if drowsy else "AWAKE"
        status_color = (0,0,255) if drowsy else (0,255,0)
        cv2.putText(frame, f"Status: {status_text}",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

        cv2.imshow("YOLOv8 Detection - Drowsiness", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Lỗi trong vòng lặp:", e)

finally:
    pygame.mixer.stop()
    pygame.mixer.quit()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()