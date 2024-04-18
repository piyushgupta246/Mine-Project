import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Specify the data directory
DATA_DIR = r"D:\Sign_language"

# Lists to store data and labels
data = []
labels = []

# Iterate through the subdirectories in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Check if dir_path is a directory
    if not os.path.isdir(dir_path):
        print(f"Skipping invalid or non-existent directory: {dir_path}")
        continue

    # Try-except block to handle any exceptions during file processing
    try:
        # Iterate through each image in the directory
        for img_filename in os.listdir(dir_path):
            # Get the full image path
            img_path = os.path.join(dir_path, img_filename)

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                # If image can't be read, skip this file
                print(f"Error reading image file: {img_path}")
                break

            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image using Mediapipe hands
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # List to store x and y coordinates
                data_aux = []
                x_ = []
                y_ = []

                # Iterate through detected hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                # Store relative coordinates
                for landmark in results.multi_hand_landmarks[0].landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Add data and labels to lists
                data.append(data_aux)
                labels.append(dir_)

    except Exception as e:
        print(f"An error occurred while processing {dir_path}: {e}")

# Save the collected data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Clean up by closing the hands module
hands.close()
print("Done")
