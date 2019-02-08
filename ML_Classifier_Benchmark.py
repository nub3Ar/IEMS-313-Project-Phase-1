from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from statistics import mean, variance
import pandas as pd
import math


############    SETUP   ############
#importing data using csv
training_data= pd.read_csv(open('Training_Dataset.csv'))
validating_data = pd.read_csv(open('Validation_Dataset.csv'))
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


############    KNN   ############
#varying number of neighbors:
for i in range(1, int(math.sqrt(dt1)),2):
#Training/Testing split:
    knn_split_accuracy = []
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_tr, y_tr)
    predicted = knn.predict(x_va)
    score = accuracy_score(y_va, predicted)
    print('Accuracy for %dnn is:' %i, score)

############    Linear Regression      ###########
print('linear')
lm = LinearRegression()
lm.fit(x_tr, y_tr)
predicted = lm.predict(x_va)
predicted = [1 if x >= 0 else -1 for x in predicted]
score = accuracy_score(y_va, predicted)
print(score)


############    Logistic Regression   ############
print('logistic')
lr = LogisticRegression(solver = 'liblinear')
lr.fit(x_tr, y_tr)
predicted = lr.predict(x_va)
predicted = [1 if x >= 0 else -1 for x in predicted]
score = accuracy_score(y_va, predicted)
print(score)
