# main1 content for image datset creation for deep learning
import os
import cv2

# Define constants
DATA_DIR = "D:/Sign_language"
number_of_classes = 26
dataset_size = 100

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open video capture")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    existing_files = [name for name in os.listdir(class_dir) if name.endswith('.jpg')]
    existing_count = len(existing_files)
    if existing_count >= dataset_size:
        print(f'Class {j} already has {existing_count} images. Skipping...')
        continue

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame for class {j}")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    counter = existing_count
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame for class {j} at counter {counter}")
            break

        cv2.imshow('frame', frame)

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

        if cv2.waitKey(25) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
