import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Improting the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into training set and test set
print(X)
"""from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test - sc_X.transform(X_test)
"""

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print(X_poly)

# Visualizing linear regression result
"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('T or B')
plt.xlabel('Position')
plt.ylabel('salary')
plt.show()
"""
# Visualizing polynomial regression result
"""plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'yellow')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('salary')
plt.show()
"""

#
print(lin_reg.predict(6.5))

print("Polynomial:")

print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))

