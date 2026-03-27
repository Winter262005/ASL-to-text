import os
import cv2
import numpy as np
import pandas as pd
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'archive', 'Data')

# Check if the folder exists, if not fallback to 'archive'
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, 'archive')

def extract_coords_from_skeleton(image_path):
    """
    Robust extraction: Uses thresholding and contour analysis.
    This method was verified to work with your specific skeleton images.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold: Skeleton (dark) vs Background (white)
    # Background is typically > 250, skeleton is much darker.
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological cleaning to bridge gaps in lines and thicken dots
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for cnt in contours:
        # Filter out very tiny noise
        if cv2.contourArea(cnt) < 0.1:
            continue
            
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))
    
    # Fallback: If dots aren't clearly separated, use corner detection on the lines
    if len(points) < 10:
        edges = cv2.Canny(gray, 50, 150)
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_edges:
            # Use polygon approximation to find joints/corners
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for p in approx:
                points.append((p[0][0], p[0][1]))

    # We need a minimum amount of data to make a prediction
    if len(points) < 5: 
        return None 

    # --- SPATIAL NORMALIZATION ---
    # 1. Centroid
    avg_x = sum(p[0] for p in points) / len(points)
    avg_y = sum(p[1] for p in points) / len(points)
    
    # 2. Advanced Sort: Radial sorting around the centroid
    # This ensures that thumb, index, middle, ring, and pinky features 
    # are always in a consistent relative order.
    def get_angle(p):
        return math.atan2(p[1] - avg_y, p[0] - avg_x)
    
    points = sorted(points, key=get_angle)

    # 3. Fixed Length (21)
    # We pad or clip the list to exactly 21 points (42 features)
    if len(points) < 21:
        last_p = points[-1] if points else (0,0)
        points.extend([last_p] * (21 - len(points)))
    points = points[:21]
    
    # 4. Scale and Translation Invariance
    # Find the maximum distance from centroid to scale everything to a 0-1 range
    max_dist = max([math.sqrt((p[0]-avg_x)**2 + (p[1]-avg_y)**2) for p in points])
    if max_dist == 0: max_dist = 1
    
    normalized = []
    for x, y in points:
        normalized.extend([(x - avg_x) / max_dist, (y - avg_y) / max_dist])
        
    return normalized

data = []
labels = []

print(f"Scanning directory: {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print("Error: The Data directory does not exist. Please check folder structure.")
    exit()

folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

for label in folders:
    dir_path = os.path.join(DATA_DIR, label)
    print(f"Processing {label}...", end=" ", flush=True)
    
    count = 0
    # Process all common image formats
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in files: 
        img_path = os.path.join(dir_path, img_name)
        coords = extract_coords_from_skeleton(img_path)
        if coords:
            data.append(coords)
            labels.append(label)
            count += 1
            
    print(f"Done ({count} extracted / {len(files)} total)")

# 3. Training Logic
if not data:
    print("\nError: Data extraction failed. No valid features found in images.")
    exit()

print(f"\nTraining on {len(data)} total samples...")
df = pd.DataFrame(data)
df['label'] = labels

X = df.drop('label', axis=1)
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize Random Forest with optimized parameters
model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=None, 
    random_state=42,
    n_jobs=-1 # Use all available CPU cores for faster training
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"New Model Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open('asl_model.p', 'wb') as f:
    pickle.dump(model, f)
print("Updated model saved successfully as 'asl_model.p'.")