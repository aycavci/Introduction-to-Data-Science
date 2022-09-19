import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss
#----------------------------------------------------------------------------#
K = 3
testSize = 0.4
randomSeed = 5
n = 10
max_depth = 19
n_estimators = 99

#Create data

data = pd.read_csv("data/featuresFlowCapAnalysis.csv")
labels = pd.read_csv("data/labelsFlowCapAnalysis.csv")


#Split the labelled data
labelled_data = data[0:178]
print("Length = ", len(labelled_data))
unlabelled_data = data[178:]

#Duplicate the 2 data
max_index = len(labels)

#Balance the data

for index, row in labels.iterrows():
    if row.item() == 2:
        labels.loc[max_index] = 2
        labelled_data.loc[max_index] = labelled_data.loc[index]
        max_index = max_index + 1

for index, row in labels.iterrows():
    if row.item() == 2:
        labels.loc[max_index] = 2
        labelled_data.loc[max_index] = labelled_data.loc[index]
        max_index = max_index + 1


for index, row in labels.iterrows():
    if row.item() == 2:
        labels.loc[max_index] = 2
        labelled_data.loc[max_index] = labelled_data.loc[index]
        max_index = max_index + 1


#train_X, validation_X = train_test_split(labelled_data, test_size=testSize, random_state = randomSeed)
#train_y, validation_y = train_test_split(labels, test_size = testSize, random_state = randomSeed)

#Balance the data




print("------------------------results----------------------------")

#---------------------------Kfold Cross validation---------------------#
kf_accuracy = 0

#kfModel = RandomForestClassifier(max_depth = max_depth, random_state = randomSeed)
kf = KFold(n_splits = n, shuffle = True, random_state = randomSeed)
i = 1


best_accuracy = 0
logloss = 0
for train_index, test_index in kf.split(labelled_data):
    print("---------------------k = ", i, "---------------------")
    i = i + 1
    train_X, train_y = labelled_data.iloc[train_index], labels.loc[train_index]
    test_X, test_y = labelled_data.iloc[test_index], labels.loc[test_index]

    #Use Smote


    kfModel = RandomForestClassifier(max_depth = max_depth, random_state = randomSeed, n_estimators = n_estimators)
    #kfModel = KNeighborsClassifier(n_neighbors = K)
    #kfModel = svm.SVC()
    train_y = train_y.squeeze()
    kfModel.fit(train_X, train_y)
    y_pred = kfModel.predict(test_X)
    kf_accuracy = kf_accuracy + accuracy_score(test_y, y_pred)
    print("Confusion matrix  = \n", confusion_matrix(test_y, y_pred))
    if accuracy_score(test_y, y_pred) > best_accuracy:
        logloss = log_loss(test_y, y_pred)
        best_train_X = train_X
        best_test_y = test_y
        best_pred_y = y_pred
        best_accuracy = accuracy_score(test_y, y_pred)
        best_depth = max_depth
        best_estimators = n_estimators
print("Log loss = ", logloss)
print("kfold accuracy score = ", kf_accuracy / i)
print("best accuracy score = ", best_accuracy)
print("Best confusion matrix = \n", confusion_matrix(best_test_y, best_pred_y))
print("Best settings are: depth = ", best_depth, " best estimators = ", best_estimators)
#-------------------------------------------------------------------#

#Check
train_X, validation_X = train_test_split(labelled_data, test_size=testSize, random_state = randomSeed)
train_y, validation_y = train_test_split(labels, test_size = testSize, random_state = randomSeed)
kfModel = RandomForestClassifier(max_depth = max_depth, random_state = randomSeed, n_estimators = n_estimators)
#kfModel = KNeighborsClassifier(n_neighbors = K)
#kfModel = svm.SVC()
train_y = train_y.squeeze()
kfModel.fit(train_X, train_y)
y_pred_unlabelled = kfModel.predict(unlabelled_data)
print("Of the unlabelled patients I found ", np.count_nonzero(y_pred_unlabelled == 2), "/ 20 of the unlabelled patients")
