import cv2
import mediapipe as mp
import pickle
import numpy as np
import math
from collections import deque

# 1. Load the trained model
try:
    model_dict = pickle.load(open('asl_model.p', 'rb'))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'asl_model.p' not found. Run train.py first.")
    exit()

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- SMOOTHING CONFIGURATION ---
prediction_window = deque(maxlen=15) # Increased window for better stability
current_display_char = "?"

cap = cv2.VideoCapture(0)
print("Starting Webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- DATA PREPROCESSING (WRIST-RELATIVE) ---
            # Landmark 0 is always the Wrist. Using it as (0,0) is more stable.
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            points = []
            for lm in hand_landmarks.landmark:
                # Subtract wrist coordinates to make the model position-invariant
                points.append((lm.x - wrist_x, lm.y - wrist_y))

            # Optional: Radial Sort (Keep this if your train.py still uses it)
            # Find local centroid of the offset points
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            
            def get_angle(p):
                return math.atan2(p[1] - avg_y, p[0] - avg_x)
            
            sorted_points = sorted(points, key=get_angle)

            # Scale Normalization
            max_dist = max([math.sqrt(p[0]**2 + p[1]**2) for p in sorted_points])
            if max_dist == 0: max_dist = 1
            
            input_features = []
            for x, y in sorted_points:
                input_features.extend([x / max_dist, y / max_dist])

            # 4. Predict
            try:
                if len(input_features) == 42:
                    probs = model_dict.predict_proba([np.asarray(input_features)])
                    max_prob = np.max(probs)
                    
                    # Log confidence to console for debugging
                    # print(f"Confidence: {max_prob:.2f}")

                    if max_prob > 0.35: # Adjusted threshold
                        prediction = model_dict.classes_[np.argmax(probs)]
                        prediction_window.append(prediction)
                    
                    if len(prediction_window) > 0:
                        current_display_char = max(set(prediction_window), key=list(prediction_window).count)

            except Exception as e:
                pass

    # UI Feedback
    cv2.rectangle(frame, (20, 20), (320, 110), (245, 117, 16), -1)
    cv2.putText(frame, "Predicted Gesture:", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, current_display_char, (30, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    cv2.imshow('ASL Real-Time Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()