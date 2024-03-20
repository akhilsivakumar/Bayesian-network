# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('score.csv')

# Extract features and target variable
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, -1].values

# Display information about the dataset
print("Dataset Information:")
print(dataset)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Initialize and train the Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = classifier.predict(X_test)

# Calculate confusion matrix and accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print confusion matrix and accuracy score
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)
