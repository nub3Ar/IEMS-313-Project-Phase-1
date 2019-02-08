from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from statistics import mean
import pandas as pd
import numpy as np
import math


############    SETUP   ############
#importing data using csv
training_dir = input("What's the directory of your training set?")
validating_dir = input("What's the directory of your validating set?")
training_data= pd.read_csv(open(training_dir))
validating_data = pd.read_csv(open(validating_dir))
#training_data= pd.read_csv(open('Training_Dataset.csv'))
#validating_data = pd.read_csv(open('Validation_Dataset.csv'))
#dimensions of the data
dt1,dt2 = training_data.shape
dv1,dv2 = validating_data.shape
#column names
col_names_tr = training_data.columns.values
col_names_va = validating_data.columns.values
#x/y variable separation
y_tr = training_data.loc[1::,'Label']
x_tr = training_data.loc[1::,'1'::]
y_va = validating_data.loc[1::,'Label']
x_va = validating_data.loc[1::,'1'::]


############      Threshold Determination     ###############
threshold_value  = []
reg = LinearRegression()
for i in range(1,dt2-1):
    reg.fit(x_tr[[str(i)]], y_tr)
    score = reg.score(x_tr[[str(i)]], y_tr)
    threshold_value.append(score)
print(np.percentile(threshold_value, 10))
threshold = np.percentile(threshold_value, 10)#(threshold_value.index(max(threshold_value))+1)*.01
if threshold >= .1:
    threshold = .1
############      Feature Selection + Logistic Regression     ###########

#determining the features with low correlation with the response
reg = LinearRegression()
R_sq = []
useful_feature = []
for i in range(1,dt2-1):
    reg.fit(x_tr[[str(i)]], y_tr)
    score = reg.score(x_tr[[str(i)]], y_tr)
    R_sq.append(score)
    if score >= threshold:
        useful_feature.append(str(i))
print(useful_feature)
#Denoising the data
new_x_tr = x_tr.loc[1::,useful_feature]
#Fitting and predicting the data
lr = LogisticRegression(solver = 'liblinear')
lr.fit(new_x_tr, y_tr)
predicted = lr.predict(x_va.loc[1::,useful_feature])
testing_accuracy = accuracy_score(y_va, predicted)
predicted = lr.predict(x_tr.loc[1::,useful_feature])
training_accuracy = accuracy_score(y_tr, predicted)
print ('The testing prediction accuracy for denoised logistic regression is:', testing_accuracy)
print ('The training prediction accuracy for denoised logistic regression is:', training_accuracy)
