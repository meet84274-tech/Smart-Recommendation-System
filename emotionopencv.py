# ==========================================
# LIVE WEBCAM EMOTION DETECTOR
# HuggingFace + OpenCV
# ==========================================

# Install:
# pip install transformers torch pillow opencv-python

import cv2
from PIL import Image
from transformers import pipeline

# Load HuggingFace emotion model
classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Convert OpenCV image (BGR → RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Predict emotion
        results = classifier(pil_image)

        # Top prediction
        emotion = results[0]['label']
        confidence = results[0]['score'] * 100

        # Display text
        text = f"{emotion} : {confidence:.2f}%"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print(e)

    # Show webcam
    cv2.imshow("Live Emotion Detector", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()