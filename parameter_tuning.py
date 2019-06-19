import datetime
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold


from feature_engineering import FeatureExtraction, BadFeatureGenerationException
from output import OutputClassification

from utils import *

RANDOM_STATE = 1
TEST_SIZE = 0.33
hz = 360
n_folds = 6

start = time.time()

output = OutputClassification("ECG Classification", "random_forest", output_dir="./output",
                              file_dir="./output/images/", random_state=RANDOM_STATE, test_size=TEST_SIZE)

output.add_info("n_processors", 12)
output.add_info("n_folds", n_folds)

csv = pd.read_csv("ecg.csv", index_col=None)

X, y = csv.drop("abnormal", axis=1), csv["abnormal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

train_start = time.time()
clf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
output.add_model_parameter("basis", clf.get_params(deep=True))

cv = StratifiedKFold(n_splits=n_folds, random_state=RANDOM_STATE)

probas = []
y_tests = []
y_preds = []

for train, test in cv.split(X, y):
    y_tests.append(y.iloc[test])
    model = clf.fit(X.iloc[train], y.iloc[train])
    proba = model.predict_proba(X.iloc[test])
    y_pred = predict_from_proba(model, proba)
    y_preds.append(y_pred)
    probas.append(proba)

make_cool_roc_curve(y_tests, probas, title="Basis parameters")

output.fill_binary_metrics(matrix_in_list(y_tests), matrix_in_list(y_preds), path="basis")

n_iter = 100

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
               }

rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=n_iter, cv=cv, verbose=2,
                               random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc")

# Fit the random search model
rf_random.fit(X_train, y_train)

best_random = rf_random.best_estimator_

best_params = best_random.get_params(deep=True)
best_params["n_iter"] = n_iter
print("Best Parameters")
print(best_params)

output.add_model_parameter("best", rf_random.best_params_)

y_tests = []
y_preds = []
probas = []
for train, test in cv.split(X, y):
    y_tests.append(y.iloc[test])
    model = best_random.fit(X.iloc[train], y.iloc[train])
    proba = best_random.predict_proba(X.iloc[test])
    y_pred = predict_from_proba(best_random, proba)
    y_preds.append(y_pred)
    probas.append(proba)

make_cool_roc_curve(y_tests, probas, True, "Best parameters")

output.fill_binary_metrics(matrix_in_list(y_tests), matrix_in_list(y_preds), path="best")

end = time.time()
output.add_performance("total_time", end - start)
output.generate_json("%s.json" % datetime.datetime.now().strftime("%y%m%d-%H%M"))
