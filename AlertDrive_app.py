import cv2
import math
import numpy as np
import dlib
import vlc
import sys
from imutils import face_utils
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# Thersholds
EYE_CLOSURE_THRESHOLD = 0.24
HEAD_POSE_DISTRACTION_THRESHOLD = 5  
DROWSINESS_FRAME_THRESHOLD_NORMAL = 15
DROWSINESS_FRAME_THRESHOLD_BODY_POSTURE = 15
DROWSINESS_FRAME_THRESHOLD_POST_YAWN = 7

# Alerts
alert = vlc.MediaPlayer('alert-sound.mp3')
focus_alert = vlc.MediaPlayer('focus.mp3')
take_a_break = vlc.MediaPlayer('take_a_break.mp3')
Suggestion = vlc.MediaPlayer('suggestions.mp3')



# counters
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1
distraction_counter = 0
distraction_cumulative = 0  
no_face_counter = 0  # Counter for frames with no face detected
yawn_counter = 0  # Yawn counter
blink_counter = 0  # Blink counter

# Initialize video capture and dlib detector
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Unable to open video capture")
    sys.exit()

avgEAR = 0.1
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
except Exception as e:
    print(f"Error loading shape_predictor_68_face_landmarks.dat: {e}")
    sys.exit()

(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

def ear(eye):
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3]))

def mar(mouth):
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (2 * euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
        shape[33],  # Nose
        shape[8],   # Chin
        shape[45],  # Left eye left corner
        shape[36],  # Right eye right corner
        shape[54],  # Left Mouth corner
        shape[48]   # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),# Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    return rotation_vector if success else None

def writeEyes(a, b, img):
    def safe_imwrite(filename, img_section):
        if img_section.size > 0:
            cv2.imwrite(filename, img_section)
    
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    safe_imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    safe_imwrite('right-eye.jpg', img[y1:y2, x1:x2])

def process_frame(frame, detector, predictor):
    global avgEAR, flag, yawn_countdown, map_flag, map_counter, distraction_counter, distraction_cumulative, no_face_counter, yawn_counter, blink_counter
    
    size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    avgEAR = 0.0

    yawn_label.config(text="")
    drowsy_label.config(text="")
    distraction_label.config(text="")
    ear_label.config(text=f"EAR: {avgEAR:.2f}")
    mar_label.config(text=f"MAR: {0:.2f}")

    if len(rects) > 0:
        no_face_counter = 0  # Reset the counter when face is detected
        
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        leftEAR = ear(leftEye)
        rightEAR = ear(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        mar_value = mar(shape[mStart:mEnd])
        mar_label.config(text=f"MAR: {mar_value:.2f}")

        if mar_value > 0.6:
            yawn_label.config(text="Yawn Detected")
            yawn_countdown = 1
            yawn_counter += 1
            yawn_counter_label.config(text=f"Yawn Count: {yawn_counter}")
            Suggestion.play()
            

        face_direction = getFaceDirection(shape, size)
        if face_direction is not None:
            pitch, yaw, roll = face_direction[0][0], face_direction[1][0], face_direction[2][0]
            if abs(pitch) > 0.5 or abs(yaw) > 0.5:
                distraction_counter += 1
                distraction_cumulative += 1
                if distraction_counter >= HEAD_POSE_DISTRACTION_THRESHOLD:
                    distraction_label.config(text="Focus on the Road! You are distracted")
                    if not focus_alert.is_playing():
                        focus_alert.play()
            else:
                distraction_counter = 0
        else:
            distraction_counter += 1
            distraction_cumulative += 1
            if distraction_counter >= HEAD_POSE_DISTRACTION_THRESHOLD:
                distraction_label.config(text="Focus on the Road! You are distracted")
                if not focus_alert.is_playing():
                    focus_alert.play()
                
        if avgEAR < EYE_CLOSURE_THRESHOLD:
            flag += 1
            eyeContourColor = (0, 255, 255)
            if yawn_countdown and flag >= DROWSINESS_FRAME_THRESHOLD_POST_YAWN:
                eyeContourColor = (147, 20, 255)
                drowsy_label.config(text="Drowsy after yawn")
                if not alert.is_playing():
                    alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1
            elif flag >= DROWSINESS_FRAME_THRESHOLD_BODY_POSTURE:
                eyeContourColor = (255, 0, 0)
                drowsy_label.config(text="Drowsy (Alert)")
                if not alert.is_playing():
                    alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1
            elif flag >= DROWSINESS_FRAME_THRESHOLD_NORMAL:
                eyeContourColor = (0, 0, 255)
                drowsy_label.config(text="Drowy")
                if not alert.is_playing():
                    alert.play()
                if map_flag:
                    map_flag = 0
                    map_counter += 1
        elif avgEAR > EYE_CLOSURE_THRESHOLD and flag:
            if flag >= 3: 
                blink_counter += 1
                blink_counter_label.config(text=f"Blink Count: {blink_counter}")
            alert.stop()
            yawn_countdown = 0
            map_flag = 1
            flag = 0

        if map_counter >= 3:
            map_flag = 1
            map_counter = 0
            if not take_a_break.is_playing():
                take_a_break.play()

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    else:
        no_face_counter += 1
        distraction_counter += 1
        distraction_cumulative += 1
        if distraction_counter >= HEAD_POSE_DISTRACTION_THRESHOLD:
            distraction_label.config(text="Focus on the Road!")
            if not focus_alert.is_playing():
                focus_alert.play()
        if no_face_counter > 10:
            distraction_label.config(text="No face detected!")
            if not focus_alert.is_playing():
                focus_alert.play()

    ear_label.config(text=f"EAR: {avgEAR:.2f}")
    return gray

def update_frame():
    ret, frame = capture.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        root.after(10, update_frame)
        return
    
    processed_frame = process_frame(frame, detector, predictor)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(processed_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Set up GUI
root = tk.Tk()
root.title("AlertDrive")

# Set background color
root.configure(bg="#2b2b2b")

# Define custom fonts
header_font = font.Font(family="Helvetica", size=17, weight="bold")
default_font = font.Font(family="Helvetica", size=17)

# Configure video label
video_label = tk.Label(root, bg="#2b2b2b")
video_label.pack()

# Configure alert labels
yawn_label = tk.Label(root, text="", font=header_font, fg="red", bg="#2b2b2b")
yawn_label.pack()

drowsy_label = tk.Label(root, text="", font=header_font, fg="red", bg="#2b2b2b")
drowsy_label.pack()

distraction_label = tk.Label(root, text="", font=header_font, fg="red", bg="#2b2b2b")
distraction_label.pack()

# Configure EAR and MAR labels
ear_label = tk.Label(root, text="EAR: 0.00", font=default_font, fg="Yellow", bg="#2b2b2b")
ear_label.pack()

mar_label = tk.Label(root, text="MAR: 0.00", font=default_font, fg="yellow", bg="#2b2b2b")
mar_label.pack()

# Configure yawn and blink counter labels
yawn_counter_label = tk.Label(root, text="Yawn Count: 0", font=default_font, fg="White", bg="#2b2b2b")
yawn_counter_label.pack()

blink_counter_label = tk.Label(root, text="Blink Count: 0", font=default_font, fg="White", bg="#2b2b2b")
blink_counter_label.pack()

# Start the video capture and processing loop
update_frame()

root.mainloop()

# Release resources
capture.release()
cv2.destroyAllWindows()


