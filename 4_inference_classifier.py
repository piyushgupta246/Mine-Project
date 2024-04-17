# program to catch and recognize to output user to give Result.

import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("C:/Users/dell/PycharmProjects/pythonProject4/model.pickle", 'rb'))
model = model_dict['model']

# Initialize video capture (use 0 for default camera, change if needed)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define gesture labels for the alphabet
labels_dict = {i: chr(65 + i) for i in range(26)}  # Maps 0-25 to 'A'-'Z'

# Main loop for real-time hand gesture recognition
while True:
    # Capture frame from the video source
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB format for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe Hands
    results = hands.process(frame_rgb)

    # Check for detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect normalized landmark coordinates for prediction
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize the coordinates
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            for landmark in hand_landmarks.landmark:
                normalized_x = (landmark.x - min_x) / (max_x - min_x)
                normalized_y = (landmark.y - min_y) / (max_y - min_y)
                data_aux.append(normalized_x)
                data_aux.append(normalized_y)

            # Check the length of data_aux
            expected_num_features = 42
            if len(data_aux) != expected_num_features:
                print(
                    f"Warning: data_aux has {len(data_aux)} features, but the model expects {expected_num_features} features.")
                continue  # Skip processing this hand if the data_aux size is incorrect

            # Convert data_aux to a NumPy array and predict gesture using the model
            input_data = np.asarray(data_aux).reshape(1, -1)  # Reshape to match expected input shape
            prediction = model.predict(input_data)

            # Retrieve the predicted character
            predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box around the hand
            x1 = int(min(x_) * frame.shape[1])
            y1 = int(min(y_) * frame.shape[0])
            x2 = int(max(x_) * frame.shape[1])
            y2 = int(max(y_) * frame.shape[0])

            # Draw bounding box and predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Add an exit condition: break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
