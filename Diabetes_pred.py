#Diabetes Prediction

#Data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[: , 0:8].values
Y = dataset.iloc[: , -1].values

#replacing 0s with NaN
X[:,1:6][X[:,1:6]==0] = np.NaN


#Taking care of missing data(NaN)

imputer = Imputer(missing_values = 'NaN' ,strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:8])
X[:,1:8]= imputer.transform(X[:,1:8])



#dividing dataset into traininsg and test set
X_train,X_test,Y_train,Y_test = train_test_split(X , Y, test_size = 0.2,random_state = 0)

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.6)

# Making the Confusion Matrix

cm = confusion_matrix(Y_test, y_pred)

#K-Fold cross validation
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)
    return classifier
classifier = KerasClassifier(build_fn =build_classifier ,batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train,y = Y_train, cv= 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#Grid Search with K-fold cross validation(CV)
def build_classifier(optimizer,batch_size,epochs,learn_rate):
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(rate = learn_rate))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = learn_rate))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn =build_classifier, verbose = 0,epochs = epochs,batch_size = batch_size)


parameters = {'batch_size' : [10,30,50],
                'epochs' : [100,200,300],
                'optimizer' : ['rmsprop','adam']
                'learn_rate' : ['0.1','0.2']}


grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
   



