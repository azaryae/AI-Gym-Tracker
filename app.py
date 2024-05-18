import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Load Mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# Function to generate video frames with pose landmarks
def generate_frames(exercise):
    cap = cv2.VideoCapture(0)
    counter_left = 0
    counter_right = 0
    stage_left = None
    stage_right = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                if exercise == 'bicep_curl':
                    # Logic for bicep curl detection
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Check if left arm is in bicep curl position
                    if left_angle < 160:  # Adjust this threshold as needed
                        stage_left = "up"
                    elif left_angle > 170 and stage_left == 'up':
                        counter_left += 1
                        stage_left = "down"

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Check if right arm is in bicep curl position
                    if right_angle < 160:  # Adjust this threshold as needed
                        stage_right = "up"
                    elif right_angle > 170 and stage_right == 'up':
                        counter_right += 1
                        stage_right = "down"
                    
                if exercise == 'lateral_raise':
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    if left_angle > 160:
                        stage_left = "down"
                    if left_angle < 90 and stage_left == 'down':
                        stage_left = "up"
                        counter_left += 1

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if right_angle > 160:
                        stage_right = "down"
                    if right_angle < 90 and stage_right == 'down':
                        stage_right = "up"
                        counter_right += 1
                
                elif exercise == 'push_up':
                    # Logic for push-up detection
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                    if left_shoulder and right_shoulder and left_elbow and right_elbow:
                        # Check if both shoulders and elbows are detected
                        if left_shoulder[1] > left_elbow[1] and right_shoulder[1] > right_elbow[1]:
                            # Check if both elbows are above shoulders
                            stage_left = "down"
                            stage_right = "down"
                        elif left_shoulder[1] < left_elbow[1] and right_shoulder[1] < right_elbow[1]:
                            # Increment count when moving from down to up position
                            if stage_left == "down":
                                counter_left += 1
                            if stage_right == "down":
                                counter_right += 1
                            stage_left = "up"
                            stage_right = "up"
            except:
                pass

            # Update reps count on the frame
            if exercise == 'bicep_curl':
                cv2.putText(image, 'REPS (Left Hand): ' + str(counter_left), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'REPS (Right Hand): ' + str(counter_right), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if exercise == 'lateral_raise':
                cv2.putText(image, 'REPS (Left Hand): ' + str(counter_left), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'REPS (Right Hand): ' + str(counter_right), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if exercise == 'push_up':
                cv2.putText(image, 'REPS (Left Side): ' + str(counter_left), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'REPS (Right Side): ' + str(counter_right), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw pose landmarks on image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Encode image to JPEG format and yield frame
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(request.args.get('exercise')), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define route for bicep curl exercise
@app.route('/bicep_curl')
def bicep_curl():
    return render_template('bicep_curl.html')

# Define route for lateral raise exercise
@app.route('/lateral_raise')
def lateral_raise():
    return render_template('lateral_raise.html')

# Define route for push up exercise
@app.route('/push_up')
def push_up():
    return render_template('push_up.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

