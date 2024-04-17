# for accuracy over the dataset read by the program 

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load data from pickle file

# Make sure the file path points to a .pickle file

data_dict = pickle.load(open("C:/Users/dell/PycharmProjects/pythonProject4/data.pickle", 'rb'))
try:
    with open('D:/Sign_language_image', 'rb') as f:
        data_dict = pickle.load(f)
except PermissionError as e:
    print(f"PermissionError: {e}")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict labels for the test set
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
# Use an alternate file path to save the model
with open('model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)
