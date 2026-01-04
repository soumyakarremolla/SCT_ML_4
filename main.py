"""
Hand Gesture Recognition - Main Script

This script will:
- Load images and labels from the LeapGestRecog dataset
- Preprocess images (resize, normalize)
- Split data into train/test sets
- (TODO) Build and train a CNN model for gesture classification
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "hand_gesture_recognition/data/leapGestRecog"
IMG_SIZE = 64  # Resize all images to 64x64

def load_data(data_dir, img_size):
    X = []
    y = []
    class_names = []
    # Each subject/session directory
    for subject in sorted(os.listdir(data_dir)):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        # Each gesture class directory
        for gesture in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            if gesture not in class_names:
                class_names.append(gesture)
            label = class_names.index(gesture)
            # Each image file
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

def main():
    print("Loading and preprocessing images...")
    X, y, class_names = load_data(DATA_DIR, IMG_SIZE)
    print(f"Loaded {len(X)} images, {len(class_names)} gesture classes.")
    print("Normalizing images and reshaping for CNN input...")
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, axis=-1)  # Add channel dimension

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Build a simple CNN model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical

    num_classes = len(class_names)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print("Building CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training CNN model...")
    model.fit(
        X_train, y_train_cat,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test_cat)
    )

    print("Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
