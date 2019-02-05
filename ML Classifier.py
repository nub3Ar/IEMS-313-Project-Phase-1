from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
knn_accuracy = []
for i in range(1, int(math.sqrt(dt1)),2):
#Training/Testing split:
    knn_split_accuracy = []
    knn = KNeighborsClassifier(n_neighbors=i)
    for j in range(2,9):
        x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=j*.1)
        knn.fit(x_train, y_train)
        predicted = knn.predict(x_test)
        score = accuracy_score(y_test, predicted)
        print('Accuracy for %dnn with %d:%d split:' %(i,j*10, 100-j*10), score)
        knn_split_accuracy.append(score)
    knn_accuracy.append(sum(knn_split_accuracy)/len(knn_split_accuracy))

print(knn_accuracy)

############    Linear Regression      ###########
lm = LinearRegression()

for j in range(2,9):
    x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=j*.1)
    lm.fit(x_train, y_train)
    predicted = lm.predict(x_test)
    predicted = [1 if x >= 0 else -1 for x in predicted]
    score = accuracy_score(y_test, predicted)
    print(score)


############    Logistic Regression   ############
lr = LogisticRegression()
lr.fit(x_tr, y_tr)
predicted = lr.predict(x_va)
score = accuracy_score(y_va, predicted)
print(score)


###########      Linear Regression      ###########
reg = LinearRegression()
R_sq = []
useful_feature = []
for i in range(1,dt2-1):
    reg.fit(x_tr.loc[1::,str(i)::], y_tr)
    score = reg.score(x_tr.loc[1::,str(i)::], y_tr)
    R_sq.append(score)
    if score >= .65:
        useful_feature.append(str(i))

new_x_tr = x_tr.loc[1::,useful_feature]



