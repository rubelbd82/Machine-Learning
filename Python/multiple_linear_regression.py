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
