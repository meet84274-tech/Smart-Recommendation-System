from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import joblib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Response
from groq import Groq


app = Flask(__name__)
camera = None
captured_frame = None

# =========================
# GROQ API
# =========================

client = Groq(
    api_key="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"
)

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODELS
# =========================

cnn_model = tf.keras.models.load_model('best_model.keras')

ml_model = joblib.load('emotion4-5-26.pkl')

le_emotion = joblib.load('le_emotion.pkl')
le_time = joblib.load('le_time.pkl')
le_pref = joblib.load('le_pref.pkl')
le_target = joblib.load('le_target.pkl')

# =========================
# EMOTION LABELS
# =========================

emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# =========================
# HAPPY CONTROL SETTINGS
# =========================

HAPPY_MIN_CONFIDENCE = 50
HAPPY_GAP_THRESHOLD = 10

# =========================
# FACE DETECTION
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# =========================
# CAMERA CONTROL
# =========================

def get_camera():

    global camera

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    return camera


def release_camera():

    global camera

    if camera is not None:
        camera.release()
        camera = None

# =========================
# USER PREFERENCE RULES
# =========================

def get_user_preference(emotion):

    rules = {
        'happy': 'video',
        'sad': 'music',
        'angry': 'article',
        'fear': 'video',
        'surprise': 'video',
        'neutral': 'music',
        'disgust': 'article'
    }

    return rules.get(emotion, 'video')

# =========================
# TIME OF DAY
# =========================

def get_time_of_day():

    hour = datetime.now().hour

    if hour >= 5 and hour < 12:
        return 'morning'

    elif hour >= 12 and hour < 16:
        return 'afternoon'

    elif hour >= 16 and hour < 19:
        return 'evening'

    else:
        return 'night'

# =========================
# FACE DETECTION
# =========================

def detect_faces_strict(gray):

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(75, 75)
    )

    return faces


def detect_faces_relaxed(gray):

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(65, 65)
    )

    return faces


def get_largest_face(faces):

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    return largest_face

# =========================
# FACE VALIDATION
# =========================

def validate_face_shape(width, height, image_width, image_height):

    face_area = width * height
    image_area = image_width * image_height

    area_ratio = face_area / image_area
    aspect_ratio = width / float(height)

    if area_ratio < 0.015:
        return False

    if aspect_ratio < 0.60 or aspect_ratio > 1.65:
        return False

    return True

# =========================
# IMAGE QUALITY CHECK
# =========================

def check_face_quality(face, width, height):

    if face is None:
        return False, 'Face could not be processed.'

    face_area = width * height

    if face_area < 5000:
        return False, 'Face detected, but it is too small.'

    brightness = np.mean(face)

    if brightness < 40:
        return False, 'Image is too dark.'

    if brightness > 235:
        return False, 'Image is too bright.'

    blur_score = cv2.Laplacian(
        face,
        cv2.CV_64F
    ).var()

    if blur_score < 40:
        return False, 'Image is blurry.'

    return True, None

# =========================
# FACE EXTRACTION
# =========================

def detect_face_from_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None, None, 'Invalid image file.'

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    image_height, image_width = gray.shape

    faces = detect_faces_strict(gray)

    if len(faces) == 0:
        faces = detect_faces_relaxed(gray)

    largest_face = get_largest_face(faces)

    if largest_face is None:
        return None, None, 'Please upload a clear human face image.'

    x, y, w, h = largest_face

    valid_shape = validate_face_shape(
        w,
        h,
        image_width,
        image_height
    )

    if valid_shape is False:
        return None, None, 'Invalid face shape detected.'

    face = gray[y:y+h, x:x+w]

    quality_status, quality_message = check_face_quality(
        face,
        w,
        h
    )

    if quality_status is False:
        return None, None, quality_message

    return face, img, None

# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_face(face):

    face = cv2.resize(
        face,
        (75, 75)
    )

    face = face / 255.0

    face = np.reshape(
        face,
        (1, 75, 75, 1)
    )

    return face

# =========================
# FINAL EMOTION LOGIC
# =========================

def select_final_emotion(prediction):

    sorted_indexes = np.argsort(prediction[0])[::-1]

    top_index = sorted_indexes[0]
    second_index = sorted_indexes[1]

    top_emotion = emotion_labels[top_index]

    top_confidence = round(
        prediction[0][top_index] * 100,
        2
    )

    second_emotion = emotion_labels[second_index]

    second_confidence = round(
        prediction[0][second_index] * 100,
        2
    )

    confidence_gap = top_confidence - second_confidence

    if top_emotion == 'happy':

        if top_confidence < HAPPY_MIN_CONFIDENCE and confidence_gap < HAPPY_GAP_THRESHOLD:
            return second_emotion, second_confidence

        if confidence_gap < HAPPY_GAP_THRESHOLD and second_confidence >= 35:
            return second_emotion, second_confidence

    return top_emotion, top_confidence

# =========================
# ML RECOMMENDATION
# =========================

def predict_recommendation_from_face(face):

    processed = preprocess_face(face)

    prediction = cnn_model.predict(processed)

    predicted_emotion, confidence = select_final_emotion(
        prediction
    )

    emotion_mapping = {
        'angry': 'anger',
        'happy': 'happiness',
        'sad': 'sadness',
        'fear': 'fear',
        'surprise': 'surprise',
        'neutral': 'neutral',
        'disgust': 'disgust'
    }

    ml_emotion = emotion_mapping.get(
        predicted_emotion,
        predicted_emotion
    )

    user_pref = get_user_preference(
        predicted_emotion
    )

    time_of_day = get_time_of_day()

    emotion_encoded = le_emotion.transform(
        [ml_emotion]
    )[0]

    pref_encoded = le_pref.transform(
        [user_pref]
    )[0]

    time_encoded = le_time.transform(
        [time_of_day]
    )[0]

    features = np.array([
        emotion_encoded,
        time_encoded,
        pref_encoded
    ]).reshape(1, -1)

    recommendation_encoded = ml_model.predict(
        features
    )[0]

    recommendation = le_target.inverse_transform(
        [recommendation_encoded]
    )[0]

    return predicted_emotion, confidence, user_pref, time_of_day, recommendation

# =========================
# GROQ API
# =========================

def generate_groq_recommendations(emotion, time_of_day, recommendation):

    api_prompt = f"""
User emotion: {emotion}
Time of day: {time_of_day}
Content type: {recommendation}

Generate exactly 10 recommendations.

Language split:
4 Hindi
3 English
3 Punjabi

Return JSON only.

Format:
{{
  "recommendations": [
    {{
      "title": "Recommendation title",
      "language": "Hindi",
      "content_type": "{recommendation}",
      "reason": "Reason"
    }}
  ]
}}
"""

    try:

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You generate mood-based recommendations in valid JSON."
                },
                {
                    "role": "user",
                    "content": api_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1500
        )

        raw_text = chat_completion.choices[0].message.content

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(cleaned)

        return parsed["recommendations"]

    except Exception as e:

        print("Groq Error:", e)

        return []

# =========================
# HOME
# =========================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# =========================
# VIDEO FEED
# =========================

def generate_frames():

    global captured_frame

    cam = get_camera()

    while True:

        success, frame = cam.read()

        if not success:
            break

        captured_frame = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detect_faces_strict(gray)

        if len(faces) == 0:
            faces = detect_faces_relaxed(gray)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            quality_status, quality_message = check_face_quality(
                face,
                w,
                h
            )

            if quality_status:

                emotion, confidence, user_pref, time_of_day, recommendation = predict_recommendation_from_face(face)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f'{emotion} ({confidence}%)',
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stop_camera', methods=['POST'])
def stop_camera():

    release_camera()

    return '', 204

# =========================
# CAPTURE
# =========================

@app.route('/capture', methods=['POST'])
def capture():

    global captured_frame

    if captured_frame is None:

        release_camera()

        return render_template(
            'webcam.html',
            error='No webcam frame available.'
        )

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        'captured.jpg'
    )

    cv2.imwrite(filepath, captured_frame)

    release_camera()

    face, original_img, error_message = detect_face_from_image(filepath)

    if error_message is not None:

        return render_template(
            'webcam.html',
            error=error_message
        )

    emotion, confidence, user_pref, time_of_day, recommendation = predict_recommendation_from_face(face)

    api_recommendations = generate_groq_recommendations(
        emotion,
        time_of_day,
        recommendation
    )

    return render_template(
        'result.html',
        image=filepath,
        emotion=emotion,
        confidence=confidence,
        preference=user_pref,
        time_of_day=time_of_day,
        recommendation=recommendation,
        api_recommendations=api_recommendations
    )

# =========================
# FILE DETECTION
# =========================

@app.route('/detect', methods=['GET', 'POST'])
def detect():

    if request.method == 'POST':

        file = request.files['image']

        if file.filename == '':

            return render_template(
                'upload.html',
                error='No file selected.'
            )

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        file.save(filepath)

        face, original_img, error_message = detect_face_from_image(filepath)

        if error_message is not None:

            return render_template(
                'upload.html',
                error=error_message
            )

        emotion, confidence, user_pref, time_of_day, recommendation = predict_recommendation_from_face(face)

        api_recommendations = generate_groq_recommendations(
            emotion,
            time_of_day,
            recommendation
        )

        return render_template(
            'result.html',
            image=filepath,
            emotion=emotion,
            confidence=confidence,
            preference=user_pref,
            time_of_day=time_of_day,
            recommendation=recommendation,
            api_recommendations=api_recommendations
        )

    return render_template('upload.html')

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(debug=True)