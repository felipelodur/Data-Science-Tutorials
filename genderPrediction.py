# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# Data for Training
# Woman's: height, weight and shoe_size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Data for Testing
_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']


#Different Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_GaussianNB = GaussianNB()

#Training the Models
clf_tree.fit(X, Y)
clf_svm.fit(X,Y)
clf_perceptron.fit(X,Y)
clf_KNN.fit(X,Y)
clf_GaussianNB.fit(X,Y)

#prediction
pred_tree = clf_tree.predict(_X)
pred_svm = clf_svm.predict(_X)
pred_per = clf_perceptron.predict(_X)
pred_KNN = clf_KNN.predict(_X)
pred_Gauss = clf_GaussianNB.predict(_X)

#accuracyCalculation
acc_tree = accuracy_score(_Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

acc_svm = accuracy_score(_Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

acc_per = accuracy_score(_Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

acc_KNN = accuracy_score(_Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

acc_Gauss = accuracy_score(_Y, pred_Gauss) * 100
print ('Accuracy for GaussNB: {}'.format(acc_Gauss))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN, acc_Gauss])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN', 3:'GaussNB'}
print('Best gender classifier is {}'.format(classifiers[index]))

