import pandas as pd
import cv2
import numpy as np
import pickle
with open('RF_final.pkl', 'rb') as f:
    m = pickle.load(f)

import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
c = 5
def print_result(result, output_image, timestamp_ms):
    global x_coord, y_coord
    # set coords when a hand is found, otherwise reset and mark absent
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        lm_list = result.hand_landmarks[0]
        for i, lm in enumerate(lm_list):
            x_coord[i] = lm.x
            y_coord[i] = lm.y

        present = True
    else:
        x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        present = False
    videoFeed(output_image, x_coord, y_coord, present)

old_pred = ''
pred = ''

def videoFeed(img, Xc, Yc, present):
    global c, old_pred, pred
    dic = {}
    n_frame = img.numpy_view()
    new_frame = np.copy(n_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
    if present:
        h, w = new_frame.shape[:2]
        x1 = int(min(Xc) * w)
        y1 = int(min(Yc) * h)
        x2 = int(max(Xc) * w)
        y2 = int(max(Yc) * h)
        cv2.rectangle(new_frame, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 255), 2)
        text = f'X: {Xc[0]:.2f}, Y: {Yc[0]:.2f}'
        cv2.putText(new_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        gray_fil = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        l = max(x2 - x1, y2 - y1) + 20
        x_start = max(cx - l//2, 0)
        y_start = max(cy - l//2, 0)
        x_end = min(cx + l//2, w)
        y_end = min(cy + l//2, h)
        cropped = gray_fil[y_start:y_end, x_start:x_end]
        resized = cv2.resize(cropped, (28, 28))
        for i in range(28):
            for j in range(28):
                dic[f'{i,j}'] = resized[i,j]/255.0
        df = pd.DataFrame(dic,index=[0])
        
        if c == 10:
            pred = m.predict(df)[0]
            old_pred = pred
            #print(pred)
            c = 0
        elif present:
            c += 1
        
        pred_show = f'Prediction: {old_pred}'
        cv2.putText(new_frame, pred_show, (x2+10, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('Live Video Feed', new_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        
cap = cv2.VideoCapture(0)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5, num_hands=1,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
    
    last_timestamp_ms = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= last_timestamp_ms:
            now_ms = last_timestamp_ms + 1
        last_timestamp_ms = now_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=gray)
        landmarker.detect_async(mp_image, last_timestamp_ms)

