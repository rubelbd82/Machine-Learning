import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Improting the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
# print(X)

y = dataset.iloc[:, 1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

# Visualize

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.scatter(X_test, y_test, color='green')

plt.show()


print(y_test)
print(y_pred)
