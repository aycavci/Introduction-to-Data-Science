import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


features = pd.read_csv("featuresFlowCapAnalysis.csv", header=None)
labels = pd.read_csv("labelsFlowCapAnalysis.csv", header=None)

train_labels = labels.iloc[:len(labels), :]
train_features = features.iloc[:len(labels), :]
test_features = features.iloc[len(labels):, :]

oversample = SMOTE(random_state=100, k_neighbors=1, n_jobs=-1)
train_features, train_labels = oversample.fit_resample(train_features, train_labels)

X_train, X_test, Y_train, Y_test = train_test_split(
													train_features,
													train_labels,
													test_size=0.1,
													shuffle=True,
													stratify=train_labels,
													random_state=100
													)

clf = GradientBoostingClassifier(random_state=100,
									n_estimators=100,
									criterion="friedman_mse",
									learning_rate=1,
									loss="deviance",
									max_features="log2",
									warm_start=True)

clf.fit(X_train, np.ravel(Y_train))

predictions = clf.predict(test_features)
print(np.count_nonzero(predictions == 2))
df = pd.DataFrame(predictions)
df.to_csv("Team_12_prediction.csv", header=False, index=False)
print(df)
