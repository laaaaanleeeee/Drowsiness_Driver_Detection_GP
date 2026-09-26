import cv2 
import mediapipe as mp 
import numpy as np 
from ultralytics import YOLO 
import time, threading 
import pygame 
import os

eyes_model = YOLO(r'D:\Graduation_Project\src\model\model_ft_yolov8-cls\result_ft_YOLOv8s_cls_eye (1)\content\runs\classify\train\weights\best.pt') 
mouth_model = YOLO(r'D:\Graduation_Project\src\model\model_ft_yolov8-cls\result_ft_YOLOv8s_cls_mouth (1)\content\runs\classify\train\weights\best.pt')

mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh( 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5 
)

LEFT_EYE_LM = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE_LM = [362, 385, 387, 263, 373, 380] 
MOUTH_LM = [61, 291, 81, 311, 13, 14, 17, 402]

def get_roi_box(landmarks, lm_indices, scale=1.4, padding=0): 
    h, w = 480, 640 
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in lm_indices]) 
    
    x1, y1 = np.min(pts, axis=0).astype(int) 
    x2, y2 = np.max(pts, axis=0).astype(int) 

    cx = (x1 + x2) // 2 
    cy = (y1 + y2) // 2 
    size = int(max((x2 - x1), (y2 - y1)) * scale) 
    
    x1 = max(0, cx - size // 2 - padding) 
    y1 = max(0, cy - size // 2 - padding) 
    x2 = min(w, cx + size // 2 + padding) 
    y2 = min(h, cy + size // 2 + padding) 
    
    return x1, y1, x2, y2

def resize_square_keep_ratio(img, out_size=96): 
    if img.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8) 
    
    h, w = img.shape[:2] 
    scale = out_size / max(h, w) 
    nh, nw = int(h * scale), int(w * scale) 
    img_resized = cv2.resize(img, (nw, nh)) 

    top = (out_size - nh) // 2 
    bottom = out_size - nh - top 
    left = (out_size - nw) // 2 
    right = out_size - nw - left 
    
    return cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

pygame.mixer.init() 
alert_sound = None 
sound_path = r"D:\Graduation_Project\src\model\DuckQuack.mp3" 

if os.path.exists(sound_path): 
    alert_sound = pygame.mixer.Sound(sound_path) 
    
def play_alert(): 
    try: alert_sound.play() 
    except: pass


EYE_CLOSED_TIME_THRESHOLD = 0.8 
YAWN_TIME_THRESHOLD = 1 
ALERT_INTERVAL = 3

eye_time = 0.0 
yawn_time = 0.0 
last_alert = 0.0

cap = cv2.VideoCapture(0) 

fps = 0.0 
prev_time = 0.0


try: 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        
        frame = cv2.resize(frame, (640, 480)) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        results = face_mesh.process(rgb) 
        
        now = time.time() 
        delta_time = now - prev_time if prev_time else 0 
        prev_time = now 
        fps = fps * 0.9 + (1/delta_time)*0.1 if delta_time > 0 else fps 
        
        drowsy = False 
        
        if results.multi_face_landmarks: 
            lm = results.multi_face_landmarks[0].landmark 
            
            lx1,ly1,lx2,ly2 = get_roi_box(lm, LEFT_EYE_LM, scale=1.3, padding=10) 
            rx1,ry1,rx2,ry2 = get_roi_box(lm, RIGHT_EYE_LM, scale=1.3, padding=10) 
            mx1,my1,mx2,my2 = get_roi_box(lm, MOUTH_LM, scale=1.4, padding=15) 

            le = resize_square_keep_ratio(frame[ly1:ly2, lx1:lx2], 96) 
            re = resize_square_keep_ratio(frame[ry1:ry2, rx1:rx2], 96) 
            mo = resize_square_keep_ratio(frame[my1:my2, mx1:mx2], 128) 
            
            eye_res = eyes_model.predict([le, re], imgsz=96, verbose=False) 
            mouth_res = mouth_model.predict(mo, imgsz=128, verbose=False)[0] 

            left_closed = (eye_res[0].probs.top1 == 0 and eye_res[0].probs.top1conf > 0.75) 
            right_closed = (eye_res[1].probs.top1 == 0 and eye_res[1].probs.top1conf > 0.75) 
            both_closed = left_closed and right_closed 

            is_yawn = (mouth_res.probs.top1 == 1 and mouth_res.probs.top1conf > 0.70) 
            
            if both_closed: 
                eye_time += delta_time 
            else: 
                eye_time = max(0.0, eye_time - delta_time) 
                
            if is_yawn: 
                yawn_time += delta_time 
            else: 
                yawn_time = max(0.0, yawn_time - delta_time) 
                
            if eye_time >= EYE_CLOSED_TIME_THRESHOLD or yawn_time >= YAWN_TIME_THRESHOLD: 
                drowsy = True 
                if now - last_alert > ALERT_INTERVAL: 
                    threading.Thread(target=play_alert, daemon=True).start() 
                    last_alert = now 
                    
            cv2.rectangle(frame, (lx1,ly1), (lx2,ly2), (0,255,0), 2) 
            cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), (0,255,0), 2) 
            cv2.rectangle(frame, (mx1,my1), (mx2,my2), (0,255,0), 2) 
            cv2.putText(frame, f"Eyes: {'CLOSED' if both_closed else 'OPEN'}", (10,30), 0,0.7,(0,255,0),2) 
            cv2.putText(frame, f"Yawn: {'YES' if is_yawn else 'NO'}", (10,60), 0,0.7,(0,255,0),2) 
            cv2.putText(frame, f"EyeTime: {eye_time:.2f}s", (10,100), 0,0.7,(255,255,0),2) 
            cv2.putText(frame, f"YawnTime: {yawn_time:.2f}s", (10,130), 0,0.7,(255,255,0),2) 
            
            if drowsy: 
                cv2.putText(frame, "DROWSY", (200,70), 0,1.8,(0,0,255),4) 

        cv2.putText(frame, f"FPS: {fps:.1f}", (540,30), 0,0.7,(0,255,255),2) 
        cv2.imshow("YOLOv8-cls Real-time", frame) 
                
        if cv2.waitKey(1) == ord('q'): 
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