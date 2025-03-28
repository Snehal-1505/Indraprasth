from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
from flask import session
from flask import Response
import mediapipe as mp
import pyttsx3
import threading
import time
import os
from flask import jsonify
import json
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # Secret key for session management

FASTAPI_URL = "http://127.0.0.1:8000/chat"  # Update with correct FastAPI URL


users_db = 'data/users.json'
workouts_db = 'data/workouts.json'
cap = cv2.VideoCapture(0)  # Initialize OpenCV webcam

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Track last feedback time
last_feedback_time = time.time()
FEEDBACK_INTERVAL = 3  # Minimum time between voice feedback (seconds)

def get_fitness_response(user_message):
    user_message = user_message.lower()
    if "workout" in user_message:
        return "Sure! Here's a simple workout plan: 1. Push-ups 2. Squats 3. Plank"
    elif "diet" in user_message:
        return "A healthy diet includes fruits, vegetables, lean proteins, and whole grains."
    elif "motivation" in user_message:
        return "You're stronger than you think! Keep pushing!"
    else:
        return "I'm here to help with fitness. Ask me about workouts, diet, or motivation!"



def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def load_users():
    if os.path.exists(users_db):
        with open(users_db, 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(users_db, 'w') as file:
        json.dump(users, file)

def speak_feedback(message):
    global last_feedback_time
    current_time = time.time()
    if current_time - last_feedback_time > FEEDBACK_INTERVAL:
        last_feedback_time = current_time
        threading.Thread(target=run_tts, args=(message,), daemon=True).start()

def run_tts(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

EXERCISES = {
    "squat": {"knee": 90, "message": "Lower your squat"},
    "pushup": {"elbow": 90, "message": "Bend your elbows more"},
    "bicep_curl": {"elbow": 45, "message": "Raise your arm higher"}
}

running = True  # Control flag for processing frames

def process_frame():
    global running, cap
    while running:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = frame.shape

            def get_point(part):
                lm = landmarks[part]
                return (int(lm.x * width), int(lm.y * height)) if lm.visibility > 0.5 else None

            shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = get_point(mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = get_point(mp_pose.PoseLandmark.LEFT_WRIST)
            hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
            knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE)
            ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE)

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)

            elbow_correct = elbow_angle and abs(elbow_angle - EXERCISES["bicep_curl"]["elbow"]) < 10
            pushup_correct = elbow_angle and abs(elbow_angle - EXERCISES["pushup"]["elbow"]) < 10
            squat_correct = knee_angle and abs(knee_angle - EXERCISES["squat"]["knee"]) < 10

            if not elbow_correct and elbow_angle:
                speak_feedback(EXERCISES["bicep_curl"]["message"])
            if not pushup_correct and elbow_angle:
                speak_feedback(EXERCISES["pushup"]["message"])
            if not squat_correct and knee_angle:
                speak_feedback(EXERCISES["squat"]["message"])

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username == "admin" and password == "password123":
            session['logged_in'] = True
            session['username'] = "admin"
            return redirect(url_for('home'))
        
        if username in users and users[username]['password'] == password:
            session['logged_in'] = False
            session['username'] = username
            return redirect(url_for('home'))

        return 'Invalid username or password'
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users:
            return 'Username already exists! Please choose another one.'

        users[username] = {'password': password}
        save_users(users)

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/home')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/surveillance')
def surveillance():
    return render_template('index.html')

@app.route('/calories')
def calories():
    return render_template('calories.html')

@app.route('/workout_planner')
def workout_planner():
    return render_template('workout_planner.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

FASTAPI_URL = "http://127.0.0.1:8000/chat"  # Update with correct FastAPI URL

@app.route('/control/<action>')
def control(action):
    global running, cap
    if action == "start":
        running = True
    elif action == "pause":
        running = False
    elif action == "stop":
        running = False
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    return jsonify({"status": action})

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True)
    except KeyboardInterrupt:
        running = False
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()