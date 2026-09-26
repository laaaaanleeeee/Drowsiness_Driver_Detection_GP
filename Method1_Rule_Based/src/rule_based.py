import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pygame
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_MAR = [61, 291, 13, 14, 81, 178, 311, 402]

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calc_ear(lm, ids):
    p = [(lm[i].x, lm[i].y) for i in ids]
    A = dist(p[1], p[5])
    B = dist(p[2], p[4])
    C = dist(p[0], p[3])
    return (A + B) / (2 * C) if C > 0 else 0

def calc_mar(lm, ids):
    p = [(lm[i].x, lm[i].y) for i in ids]
    A = dist(p[2], p[3])
    B = dist(p[4], p[5])
    C = dist(p[6], p[7])
    D = dist(p[0], p[1])
    return (A + B + C) / (2 * D) if D > 0 else 0

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.7
EYE_CLOSED_TIME_THRESHOLD = 1
YAWN_TIME_THRESHOLD = 1.5
ALERT_INTERVAL = 3

# url = "http://192.168.10.102:8080/video"

# cap = VideoStream(url)

# url = "rtsp://admin:admin@192.168.10.102:8554/stream"
# cap = cv2.VideoCapture(url)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"D:\Graduation_Project\src\evaluation\test_video\7-FemaleNoGlasses.avi")
# cap = cv2.VideoCapture(r"D:\Graduation_Project\src\evaluation\test_video\14-MaleNoGlasses.avi")

# cap = cv2.VideoCapture(r"C:\Users\Admin\Pictures\Camera Roll\WIN_20251207_19_11_33_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_07_46_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_09_14_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_10_54_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_11_50_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_13_05_Pro.mp4")
# cap = cv2.VideoCapture("C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20251130_22_14_55_Pro.mp4")


def play_alert():
    try:
        if alert_sound:
            alert_sound.play()
    except Exception as e:
        print(f"Alert error: {e}")

pygame.mixer.init()
alert_sound = None
sound_path = r"D:\Graduation_Project\src\model\DuckQuack.mp3"

if os.path.exists(sound_path):
    try:
        alert_sound = pygame.mixer.Sound(sound_path)
    except Exception as e:
        print(f"Không load được âm thanh: {e}")
else:
    print(f"File âm thanh không tồn tại: {sound_path}")


eye_closed_time = 0
yawn_time = 0
last_alert = 0
prev_time = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame!")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        h, w = frame.shape[:2]

        now = time.time()
        delta_time = now - prev_time if prev_time else 0
        fps = 1 / delta_time if delta_time > 0 else 0
        prev_time = now

        drowsy = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_ear = calc_ear(lm, LEFT_EYE)
            right_ear = calc_ear(lm, RIGHT_EYE)
            
            mar = calc_mar(lm, MOUTH_MAR)

            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                eye_closed_time += delta_time
            else:
                eye_closed_time = 0
                # eye_closed_time = max(0, eye_closed_time - delta_time)

            if mar > MAR_THRESHOLD:
                yawn_time += delta_time
            else:
                yawn_time = 0
                # yawn_time = max(0, yawn_time - delta_time)

            if (eye_closed_time >= EYE_CLOSED_TIME_THRESHOLD or
                yawn_time >= YAWN_TIME_THRESHOLD):
                drowsy = True
                if now - last_alert > ALERT_INTERVAL:
                    threading.Thread(target=play_alert, daemon=True).start()
                    last_alert = now

            cv2.putText(frame, f"LEFT EYE: {left_ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"RIGHT EYE: {right_ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.putText(frame, f"Eye Closed Time: {eye_closed_time:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Yawn Time: {yawn_time:.2f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            eye_closed_time = 0
            yawn_time = 0

        status_text = 'DROWSY' if drowsy else 'AWAKE'
        status_color = (0, 0, 255) if drowsy else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (10, 440), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"FPS: {fps:.1f}", (500, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    print(f"Lỗi trong vòng lặp: {e}")
finally:
    pygame.mixer.stop()
    pygame.mixer.quit()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()