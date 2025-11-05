# Importing required libraries
import numpy as np  # Used for numerical operations and handling arrays
import pandas as pd  # Used for data manipulation and analysis
from sklearn.model_selection import (
    train_test_split,
)  # Used to split the dataset into training and testing sets
from sklearn.linear_model import (
    LogisticRegression,
)  # Used for building the logistic regression model
from sklearn.metrics import (
    accuracy_score,
)  # Used to measure the accuracy of predictions

# Loading the dataset into a pandas DataFrame
sonar_data = pd.read_csv(
    "/content/sonar_data.csv", header=None
)  # Reads the CSV file without column headers

# Displaying the first 5 rows of the dataset
sonar_data.head()  # Gives a quick look at the structure and contents of the dataset

# Displaying the shape of the dataset
sonar_data.shape  # Returns the number of rows and columns in the dataset (208, 61)

# Getting statistical information about the dataset
sonar_data.describe()  # Shows mean, standard deviation, min, max, and quartiles for each column

# Displaying the count of each label (Rock or Mine)
sonar_data[
    60
].value_counts()  # Counts how many samples belong to each category: 'R' or 'M'

# Calculating mean values for each column grouped by label
sonar_data.groupby(
    60
).mean()  # Gives average feature values separately for 'R' and 'M' categories

# Splitting data into features (X) and labels (Y)
X = sonar_data.drop(
    columns=60, axis=1
)  # Drops the label column (index 60) to get only feature data
Y = sonar_data[60]  # Extracts the label column for classification targets

# Printing the features and labels
print(X)  # Displays all input features (numerical data from sonar readings)
print(Y)  # Displays the corresponding labels ('R' for Rock, 'M' for Mine)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)  # 90% training data, 10% test data, stratified by class distribution

# Displaying dataset shapes after splitting
print(
    X.shape, X_train.shape, X_test.shape
)  # Prints sizes of total, training, and testing data respectively
print(X_train)  # Prints training features
print(Y_train)  # Prints corresponding training labels

# Creating a Logistic Regression model
model = LogisticRegression()  # Initializes logistic regression classifier

# Training the model with training data
model.fit(
    X_train, Y_train
)  # Fits the model using training data (learns weights and bias)

# Predicting labels for training data
X_train_prediction = model.predict(
    X_train
)  # Generates predictions for the same data used to train the model

# Calculating training accuracy
training_data_accuracy = accuracy_score(
    X_train_prediction, Y_train
)  # Compares predictions with actual labels
print(
    "Accuracy on training data: ", training_data_accuracy
)  # Displays how well the model fits training data

# Predicting labels for test data
X_test_prediction = model.predict(X_test)  # Generates predictions for unseen test data

# Calculating test accuracy
test_data_accuracy = accuracy_score(
    X_test_prediction, Y_test
)  # Measures model performance on unseen data
print("Accuracy on test data: ", test_data_accuracy)  # Displays the test accuracy score

# Making a prediction on new data
input_data = (
    0.0286,
    0.0453,
    0.0277,
    0.0174,
    0.0384,
    0.0990,
    0.1201,
    0.1833,
    0.2105,
    0.3039,
    0.2988,
    0.4250,
    0.6343,
    0.8198,
    1.0000,
    0.9988,
    0.9508,
    0.9025,
    0.7234,
    0.5122,
    0.2074,
    0.3985,
    0.5890,
    0.2872,
    0.2043,
    0.5782,
    0.5389,
    0.3750,
    0.3411,
    0.5067,
    0.5580,
    0.4778,
    0.3299,
    0.2198,
    0.1407,
    0.2856,
    0.3807,
    0.4158,
    0.4054,
    0.3296,
    0.2707,
    0.2650,
    0.0723,
    0.1238,
    0.1192,
    0.1089,
    0.0623,
    0.0494,
    0.0264,
    0.0081,
    0.0104,
    0.0045,
    0.0014,
    0.0038,
    0.0013,
    0.0089,
    0.0057,
    0.0027,
    0.0051,
    0.0062,
)
# Tuple containing sonar readings for one object to classify

# Converting the input data to a numpy array
input_data_as_numpy_array = np.asarray(
    input_data
)  # Converts tuple into a numpy array for model input

# Reshaping the input for prediction (1 sample with multiple features)
input_data_reshaped = input_data_as_numpy_array.reshape(
    1, -1
)  # Reshapes to 2D array as model expects (1 row, 60 columns)

# Making prediction using trained model
prediction = model.predict(
    input_data_reshaped
)  # Predicts whether the object is Rock ('R') or Mine ('M')

# Displaying the raw model output
print(prediction)  # Prints predicted label ('R' or 'M')

# Interpreting and displaying the result in plain text
if prediction[0] == "R":  # If prediction is 'R'
    print("The object is a rock")  # Prints rock classification
else:
    print("The object is a mine")  # Prints mine classification
