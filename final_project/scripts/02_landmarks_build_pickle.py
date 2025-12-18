"""
Extract MediaPipe hand landmarks from images and save as pickle.
Used for KNN, Logistic Regression, and Decision Tree models.
"""

import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

IMG_EXTS = (".jpg", ".jpeg", ".png")

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]

    x_min, y_min = min(xs), min(ys)

    features = []
    for lm in hand.landmark:
        features.extend([lm.x - x_min, lm.y - y_min])

    return features if len(features) == 42 else None

def load_dataset(root):
    X, y = [], []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue

        label = cls.lower()
        for f in os.listdir(cls_dir):
            if f.lower().endswith(IMG_EXTS):
                feat = extract_features(os.path.join(cls_dir, f))
                if feat is not None:
                    X.append(feat)
                    y.append(label)

    return X, y

def main():
    train_dir = "asl_dataset/asl_dataset"
    unseen_dir = "data"

    X_train, y_train = load_dataset(train_dir)
    X_unseen, y_unseen = load_dataset(unseen_dir)

    with open("models/classical/landmarks_train_unseen.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_unseen": X_unseen,
            "y_unseen": y_unseen
        }, f)

    print("Saved landmark features to models/classical/")

if __name__ == "__main__":
    main()
