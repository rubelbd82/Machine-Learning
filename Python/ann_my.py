# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])

labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Make the Artificial Neural Network!

# Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Install the ANN
classifier = Sequential()

#Add input layer and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

#Add second layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#Add output layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # what is loss and accuracy

#Fit the ANN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Make the prediction and evaluate the model

# Predict test set reuslt
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # why exit 0.5 ?

# Make Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
