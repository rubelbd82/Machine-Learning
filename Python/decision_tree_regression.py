import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Improting the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into training set and test set

"""from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test - sc_X.transform(X_test)
"""


# Fitting the DecisionTree Model to the data set
# DecisionTreeModel

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)


#Predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the Regression results
"""plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position')
plt.ylabel('salary')
plt.show()"""

# Visualizing the Regression result in high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(np.reshape(a=X_grid, newshape=(-1, 1))), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position')
plt.ylabel('salary')
plt.show()
