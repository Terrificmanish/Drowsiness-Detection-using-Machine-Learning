from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import dlib
import base64
from scipy.spatial import distance as dist
from collections import deque
import os
from datetime import datetime

web_app = Flask(__name__)
SNAPSHOT_DIR = 'snapshots'
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
    
SCALE_RATIO = 1.5
FRAME_HEIGHT = 460
EAR_THRESHOLD = 0.27
MODEL_FILE_PATH = "projects/models/shape_predictor_70_face_landmarks.dat"
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
BLINK_DURATION = 0.15
DROWSY_DURATION = 1.5
EAR_BUFFER_LENGTH = 5
DROWSY_FRAME_COUNT = 15
GAMMA_VALUE = 1.5
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(MODEL_FILE_PATH)

inverse_gamma = 1.0 / GAMMA_VALUE
gamma_table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in range(0, 256)]).astype("uint8")
ear_values = deque(maxlen=EAR_BUFFER_LENGTH)
consecutive_drowsy_frames = 0
calibration_in_progress = True
calibration_frames = []
calibration_threshold = None

def apply_gamma_correction(image):
    return cv2.LUT(image, gamma_table)

def equalize_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_landmarks(image):
    resized_image = cv2.resize(image, None,
                               fx=1.0 / SCALE_RATIO,
                               fy=1.0 / SCALE_RATIO,
                               interpolation=cv2.INTER_LINEAR)

    detected_faces = face_detector(resized_image, 0)
    if len(detected_faces) == 0:
        return 0

    scaled_face = dlib.rectangle(int(detected_faces[0].left() * SCALE_RATIO),
                                 int(detected_faces[0].top() * SCALE_RATIO),
                                 int(detected_faces[0].right() * SCALE_RATIO),
                                 int(detected_faces[0].bottom() * SCALE_RATIO))

    landmarks = []
    [landmarks.append((point.x, point.y)) for point in shape_predictor(image, scaled_face).parts()]
    return landmarks

def analyze_drowsiness(image):
    global consecutive_drowsy_frames

    adjusted_image = apply_gamma_correction(image)
    adjusted_image = equalize_histogram(adjusted_image)

    landmarks = detect_landmarks(adjusted_image)
    if landmarks == 0:
        return "Face Not Found"

    left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
    right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]

    left_ear = calculate_ear(np.array(left_eye))
    right_ear = calculate_ear(np.array(right_eye))
    ear = (left_ear + right_ear) / 2.0

    ear_values.append(ear)
    avg_ear = np.mean(ear_values)

    if avg_ear < EAR_THRESHOLD:
        consecutive_drowsy_frames += 1
    else:
        consecutive_drowsy_frames = 0

    if consecutive_drowsy_frames >= DROWSY_FRAME_COUNT:
        return "Alert"
    
    return "No Drowsiness"

def calibrate_threshold(image):
    global calibration_frames, calibration_threshold, calibration_in_progress

    adjusted_image = apply_gamma_correction(image)
    adjusted_image = equalize_histogram(adjusted_image)

    landmarks = detect_landmarks(adjusted_image)
    if landmarks == 0:
        return "Face Not Found"

    left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
    right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]

    left_ear = calculate_ear(np.array(left_eye))
    right_ear = calculate_ear(np.array(right_eye))
    ear = (left_ear + right_ear) / 2.0

    calibration_frames.append(ear)
    if len(calibration_frames) >= 30:
        calibration_threshold = np.mean(calibration_frames)
        calibration_in_progress = False
        return f"Calibration Complete. Threshold set to {calibration_threshold:.2f}"

    return "Calibrating..."

def save_video(frames, output_file='output_video.avi'):
    height, width = frames[0].shape[:2]
    vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
    for frame in frames:
        vid_writer.write(frame)
    vid_writer.release()

def save_snapshot(frame):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(SNAPSHOT_DIR, f'snapshot_{timestamp}.jpg')
    cv2.imwrite(filename, frame)

@web_app.route('/')
def home():
    return render_template('index.html')

@web_app.route('/process_video', methods=['POST'])
def process_video():
    global calibration_in_progress, calibration_threshold

    try:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]
        decoded_image = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'prediction': 'Image decoding failed'}), 400

        if calibration_in_progress:
            message = calibrate_threshold(frame)
            return jsonify({'prediction': message})
        else:
            global EAR_THRESHOLD
            EAR_THRESHOLD = calibration_threshold if calibration_threshold is not None else EAR_THRESHOLD
            prediction = analyze_drowsiness(frame)
            
            if prediction == "Alert":
                save_snapshot(frame)

            return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'prediction': 'Error processing video frame', 'error': str(e)}), 500

@web_app.route('/save_video', methods=['POST'])
def save_video_route():
    global calibration_in_progress, calibration_threshold

    try:
        video_frames = request.json['frames']
        frames = [cv2.imdecode(np.frombuffer(base64.b64decode(frame), np.uint8), cv2.IMREAD_COLOR) for frame in video_frames]
        
        if not frames:
            return jsonify({'status': 'No frames to save'}), 400
        
        save_video(frames)
        return jsonify({'status': 'Video saved successfully'})

    except Exception as e:
        return jsonify({'status': 'Error saving video', 'error': str(e)}), 500

if __name__ == '__main__':
    web_app.run(debug=True)
