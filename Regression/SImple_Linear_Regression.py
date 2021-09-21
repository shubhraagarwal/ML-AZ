import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
# ! Importing the dataset

dataset = pd.read_csv("Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ! Splitting the dataset into training and test set
# * X_train contains the independent varibale
#*  y_train contains the dependent variable 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# ! Training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ! Predicting the test set results

y_pred = regressor.predict(X_test)

#! Visualing the training set results
plt.scatter(X_train, y_train, color= "red")
plt.plot(X_train, regressor.predict(X_train), color="blue" )
plt.title("Salary vs Experience(train)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#! Visualing the test set results
plt.scatter(X_test, y_test, color= "red")
#* Here we are not replacing X_train with X_test because this line tells us about the data predicted and how close our results are to the training set
plt.plot(X_train, regressor.predict(X_train), color="blue" )
plt.title("Salary vs Experience(test)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

