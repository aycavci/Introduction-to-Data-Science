import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

DT = DecisionTreeClassifier(random_state=100)
DTestimators = {
				"criterion": ["gini", "entropy"],
				"splitter": ["best", "random"],
				"max_features": ["auto", "sqrt", "log2", None],
				"class_weight": ["balanced", None]
				}

AdaB = AdaBoostClassifier(random_state=100)
AdaBestimators = {
				"n_estimators": [10, 50, 100, 200],
				"algorithm": ["SAMME", "SAMME.R"],
				"learning_rate": [0.01, 0.1, 1]
				}

GDB = GradientBoostingClassifier(random_state=100)
GDBestimators = {
				"n_estimators": [10, 100, 200],
				"learning_rate": [0.01, 0.1, 1],
				"loss": ["deviance", "exponential"],
				"criterion": ["friedman_mse", "mse", "mae"],
				"max_features": ["auto", "sqrt", "log2", None],
				"warm_start": [True, False]
				}

experiments = {
				"KFoldCV": [],
				"BestHyperParameters": [],
				"TrainAccuracy": [],
				"TestAccuracy": [],
				"LogLoss": [],
				"TrainConfusionMatrix": [],
				"TestConfusionMatrix": [],
				"UnlabelledClass1": [],
				"UnlabelledClass2": []
				}

features = pd.read_csv("featuresFlowCapAnalysis.csv", header=None)
labels = pd.read_csv("labelsFlowCapAnalysis.csv", header=None)


train_labels = labels.iloc[:len(labels), :]
train_features = features.iloc[:len(labels), :]
test_features = features.iloc[len(labels):, :]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

counter = Counter(np.ravel(train_labels))
for label, _ in counter.items():
	row_ix = np.where(np.ravel(train_labels) == label)[0]
	axes[0].scatter(train_features.iloc[row_ix, 0], train_features.iloc[row_ix, 1], label=str(label))
axes[0].title.set_text("Features Scatter Plot Without Preprocessing")
axes[0].legend()

oversample = SMOTE(random_state=100, n_jobs=-1)
train_features, train_labels = oversample.fit_resample(train_features, train_labels)

X_train, X_test, Y_train, Y_test = train_test_split(
													train_features,
													train_labels,
													test_size=0.1,
													shuffle=True,
													stratify=train_labels,
													random_state=100
													)

counter = Counter(np.ravel(train_labels))
for label, _ in counter.items():
	row_ix = np.where(np.ravel(train_labels) == label)[0]
	axes[1].scatter(train_features.iloc[row_ix, 0], train_features.iloc[row_ix, 1], label=str(label))
axes[1].title.set_text("Features Scatter Plot With SMOTE Over Sampling Minority Class")
axes[1].legend()

fig.tight_layout()
plt.show()


train_labels = labels.iloc[:len(labels), :]
train_features = features.iloc[:len(labels), :]
test_features = features.iloc[len(labels):, :]

X_train, X_test, Y_train, Y_test = train_test_split(
													train_features,
													train_labels,
													test_size=0.1,
													shuffle=True,
													stratify=train_labels,
													random_state=100
													)

for y in range(2, 11):
	print("CV: ", y)
	experiments["KFoldCV"].append(y)

	clf = GridSearchCV(DT, param_grid=DTestimators, n_jobs=-1, cv=y, refit=True, return_train_score=True, scoring="neg_log_loss")

	clf.fit(X_train, np.ravel(Y_train))

	print("Best Parameters; ", clf.best_params_)
	experiments["BestHyperParameters"].append(clf.best_params_)

	print("Train Accuracy: ", accuracy_score(Y_train, clf.predict(X_train)))
	experiments["TrainAccuracy"].append(accuracy_score(Y_train, clf.predict(X_train)))

	print("Test Accuracy: ", accuracy_score(Y_test, clf.predict(X_test)))
	experiments["TestAccuracy"].append(accuracy_score(Y_test, clf.predict(X_test)))

	print("Log Loss: ", log_loss(Y_test, clf.predict(X_test)))
	experiments["LogLoss"].append(log_loss(Y_test, clf.predict(X_test)))

	cm_train = confusion_matrix(Y_train, clf.predict(X_train))
	cm_test = confusion_matrix(Y_test, clf.predict(X_test))
	print(cm_train)
	print(cm_test)
	experiments["TrainConfusionMatrix"].append(cm_train)
	experiments["TestConfusionMatrix"].append(cm_test)

	count = clf.predict(test_features)
	experiments["UnlabelledClass1"].append(np.count_nonzero(count == 1))
	experiments["UnlabelledClass2"].append(np.count_nonzero(count == 2))

	print(" ")

df = pd.DataFrame.from_dict(experiments)
df = df.sort_values(by="TestAccuracy", ascending=False)
# df.to_csv("GDBNoPre.csv")
print(df)


experiments = {
				"SmoteKNeighbours": [],
				"KFoldCV": [],
				"BestHyperParameters": [],
				"TrainAccuracy": [],
				"TestAccuracy": [],
				"LogLoss": [],
				"TrainConfusionMatrix": [],
				"TestConfusionMatrix": [],
				"UnlabelledClass1": [],
				"UnlabelledClass2": []
				}

for x in range(1, 10):
	print("SmoteKNeighbours: ", x)

	oversample = SMOTE(random_state=100, k_neighbors=x, n_jobs=-1)
	train_features, train_labels = oversample.fit_resample(train_features, train_labels)

	X_train, X_test, Y_train, Y_test = train_test_split(
														train_features,
														train_labels,
														test_size=0.1,
														shuffle=True,
														stratify=train_labels,
														random_state=100
														)

	for y in range(2, 11):
		print("CV: ", y)
		experiments["SmoteKNeighbours"].append(x)
		experiments["KFoldCV"].append(y)

		clf = GridSearchCV(GDB, param_grid=GDBestimators, n_jobs=-1, cv=y, refit=True, return_train_score=True, scoring="neg_log_loss")

		clf.fit(X_train, np.ravel(Y_train))

		print("Best Parameters; ", clf.best_params_)
		experiments["BestHyperParameters"].append(clf.best_params_)

		print("Train Accuracy: ", accuracy_score(Y_train, clf.predict(X_train)))
		experiments["TrainAccuracy"].append(accuracy_score(Y_train, clf.predict(X_train)))

		print("Test Accuracy: ", accuracy_score(Y_test, clf.predict(X_test)))
		experiments["TestAccuracy"].append(accuracy_score(Y_test, clf.predict(X_test)))

		print("Log Loss: ", log_loss(Y_test, clf.predict(X_test)))
		experiments["LogLoss"].append(log_loss(Y_test, clf.predict(X_test)))

		cm_train = confusion_matrix(Y_train, clf.predict(X_train))
		cm_test = confusion_matrix(Y_test, clf.predict(X_test))
		print(cm_train)
		print(cm_test)
		experiments["TrainConfusionMatrix"].append(cm_train)
		experiments["TestConfusionMatrix"].append(cm_test)

		count = clf.predict(test_features)
		experiments["UnlabelledClass1"].append(np.count_nonzero(count == 1))
		experiments["UnlabelledClass2"].append(np.count_nonzero(count == 2))

		print(" ")

df = pd.DataFrame.from_dict(experiments)
df = df.sort_values(by="TestAccuracy", ascending=False)
# df.to_csv("GDBSmote.csv")
print(df)
