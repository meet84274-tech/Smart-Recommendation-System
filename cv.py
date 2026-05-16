import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================
model = load_model("Emotion.h5")

emotion_labels = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprises"
]

# =====================================
# START WEBCAM
# =====================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# =====================================
# LOOP
# =====================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =================================
    # PREPROCESS IMAGE
    # =================================
    img = cv2.resize(frame, (150, 150))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # =================================
    # PREDICTION
    # =================================
    preds = model.predict(img, verbose=0)

    emotion_index = np.argmax(preds)

    confidence = np.max(preds) * 100

    emotion = emotion_labels[emotion_index]

    # =================================
    # DISPLAY TEXT
    # =================================
    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # =================================
    # SHOW WINDOW
    # =================================
    cv2.imshow("Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# RELEASE
# =====================================
cap.release()
cv2.destroyAllWindows()