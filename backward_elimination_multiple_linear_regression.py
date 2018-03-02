import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Improting the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
# print(X)

y = dataset.iloc[:, 4].values


#
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelEncoder_X = LabelEncoder()
# X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()
# labelEncoder_Y = LabelEncoder()
# y = labelEncoder_Y.fit_transform(y)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Remove dummy variable error
X = X[:, 1:]


# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test - sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)


# print(y_test)


import statsmodels.formula.api as sm

# Instead of using hardcoded line number 50, I used X.shape[0] to get number of rows of matrix X
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)  # axis, 0 = row, 1 = column

print(X)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())  # I'm using pycharm, so I printed it in console.


X_opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())


X_opt = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

