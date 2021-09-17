import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# ! Importing the dataset

dataset = pd.read_csv("../Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ! Splitting the dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# ! Training the simple linear regression model on the training set

