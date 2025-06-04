import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

click_delay = 1  # seconds
last_click_time = 0

def run_virtual_mouse():
    global last_click_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                index_finger_up = landmarks[8][1] < landmarks[6][1]
                middle_finger_up = landmarks[12][1] < landmarks[10][1]

                cursor_x = int(landmarks[8][0] * screen_width)
                cursor_y = int(landmarks[8][1] * screen_height)

                # Move cursor to index finger tip
                pyautogui.moveTo(cursor_x, cursor_y)

                # Double click if only index finger is up
                if index_finger_up and not middle_finger_up:
                    current_time = time.time()
                    if current_time - last_click_time > click_delay:
                        pyautogui.doubleClick()
                        last_click_time = current_time

        cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_virtual_mouse()
