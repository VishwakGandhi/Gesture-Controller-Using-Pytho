#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pygetwindow pyautogui


# In[6]:


import cv2
from mediapipe.python import solutions as mp
import numpy as np
from math import hypot
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mpHands = mp.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

# Initialize screen brightness control
monitor = sbc.Monitor(0)  # Adjust monitor index if needed

# Function to get the volume interface
def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

# Initialize OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize variables for brightness control
brightness_length_buffer = []
brightness_previous_length = 0
brightness_previous_x2, brightness_previous_y2 = 0, 0

# Initialize variables for volume control
volume_length_buffer = []
volume_previous_length = 0
volume_previous_x2, volume_previous_y2 = 0, 0

while True:
    # Read a frame from the camera
    ret, img = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    brightness_lmList = []
    volume_lmList = []

    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            fingers_open = [handlandmark.landmark[i].y < handlandmark.landmark[i - 1].y for i in range(5, 1, -1)]

            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Draw landmarks for both brightness and volume
                brightness_lmList.append([cx, cy])
                volume_lmList.append([cx, cy])
                cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

            # Brightness control based on the movement of the index finger (Left Hand)
            if not all(fingers_open[1:]):
                x1, y1 = brightness_lmList[4][0], brightness_lmList[4][1]
                x2, y2 = brightness_lmList[8][0], brightness_lmList[8][1]
                brightness_length = hypot(x2 - x1, y2 - y1)

                # Apply filters to avoid false touches
                brightness_threshold = 15
                if brightness_length > brightness_threshold:
                    brightness_length_buffer.append(brightness_length)

                    # Use a simple moving average for length
                    brightness_buffer_size = 5
                    if len(brightness_length_buffer) > brightness_buffer_size:
                        brightness_length = sum(brightness_length_buffer[-brightness_buffer_size:]) / brightness_buffer_size

                        # Introduce dead zone
                        brightness_dead_zone = 5
                        if abs(brightness_length - brightness_previous_length) > brightness_dead_zone:
                            # Process the gesture

                            # Calculate velocity
                            brightness_velocity = hypot((x2 - x1) - (x2 - brightness_previous_x2), (y2 - y1) - (y2 - brightness_previous_y2))

                            # Check if velocity is above a certain threshold
                            brightness_velocity_threshold = 10
                            if brightness_velocity > brightness_velocity_threshold:
                                # Process the gesture
                                brightness_value = np.interp(brightness_length, [15, 220], [0, 100])
                                print("Brightness:", brightness_value)
                                monitor.set_brightness(int(brightness_value))

                        brightness_previous_length = brightness_length
                        brightness_previous_x2, brightness_previous_y2 = x2, y2

            # Volume control based on the movement of the index finger (Right Hand)
            elif all(fingers_open):
                x1, y1 = volume_lmList[4][0], volume_lmList[4][1]
                x2, y2 = volume_lmList[8][0], volume_lmList[8][1]
                volume_length = hypot(x2 - x1, y2 - y1)

                # Apply filters to avoid false touches
                volume_threshold = 15
                if volume_length > volume_threshold:
                    volume_length_buffer.append(volume_length)

                    # Use a simple moving average for length
                    volume_buffer_size = 5
                    if len(volume_length_buffer) > volume_buffer_size:
                        volume_length = sum(volume_length_buffer[-volume_buffer_size:]) / volume_buffer_size

                        # Introduce dead zone
                        volume_dead_zone = 5
                        if abs(volume_length - volume_previous_length) > volume_dead_zone:
                            # Process the gesture

                            # Calculate velocity
                            volume_velocity = hypot((x2 - x1) - (x2 - volume_previous_x2), (y2 - y1) - (y2 - volume_previous_y2))

                            # Check if velocity is above a certain threshold
                            volume_velocity_threshold = 10
                            if volume_velocity > volume_velocity_threshold:
                                # Process the gesture
                                volume_value = np.interp(volume_length, [15, 220], [0, 1])
                                print("Volume:", volume_value)
                                volume_interface = get_volume_interface()
                                volume_interface.SetMasterVolumeLevelScalar(volume_value, None)

                            volume_previous_length = volume_length
                            volume_previous_x2, volume_previous_y2 = x2, y2

    cv2.imshow('Image', img)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




