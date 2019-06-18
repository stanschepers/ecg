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


from feature_extraction import FeatureExtraction, BadFeatureGenerationException
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

# negative_path = "data/MLII/1 NSR"
# positive_paths = [None, 'data/MLII/1 NSR', 'data/MLII/2 APB',
#                   'data/MLII/3 AFL', 'data/MLII/4 AFIB', 'data/MLII/5 SVTA', 'data/MLII/6 WPW', 'data/MLII/7 PVC',
#                   'data/MLII/8 Bigeminy', 'data/MLII/9 Trigeminy', 'data/MLII/10 VT', 'data/MLII/11 IVR',
#                   'data/MLII/12 VFL', 'data/MLII/13 Fusion',
#                   'data/MLII/14 LBBBB', 'data/MLII/15 RBBBB', 'data/MLII/16 SDHB', 'data/MLII/17 PR']
# """
# Name    |  #
# 1 NSR | 283
# 2 APB | 66
# 3 AFL | 20
# 4 AFIB | 135
# 5 SVTA | 13
# 6 WPW | 21
# 7 PVC | 133
# 8 Bigeminy | 55
# 9 Trigeminy | 13
# 10 VT | 10
# 11 IVR | 10
# 12 VFL | 10
# 13 Fusion | 11
# 14 LBBBB | 103
# 15 RBBBB | 62
# 16 SDHB | 10
# 17 PR | 45
# """
#
# positive_indices = [2, 4, 7, 8, 14, 15, 7, 16, 17]
#
# X = []
# y = []
# features_names = []
# total_len = 0
#
#
# total_positive = 0
# bad_positive = 0
# for i in positive_indices:
#     for file in os.listdir(positive_paths[i]):
#         ecg = io.matlab.loadmat(positive_paths[i] + "/" + file)["val"][0]
#         try:
#             features, features_names = FeatureExtraction.generate_features(ecg, hz)
#             features.append(1)
#             features_names.append("Abnormal")
#             X.append(features)
#             y.append(1)
#         except BadFeatureGenerationException:
#             print(f"1: {file} [{positive_paths[i]}]")
#             bad_positive += 1
#         total_positive += 1
#     len_dir = len(os.listdir(positive_paths[i]))
#     total_len += len_dir
# print("# positief", total_len)
#
# all_negative_cases = os.listdir(negative_path)
# negative_cases = all_negative_cases
# # negative_cases = random.sample(all_negative_cases, total_len) if total_len < len(
# #     os.listdir(negative_path)) else all_negative_cases
#
#
# print("# negatief", len(negative_cases))
#
# output.add_feature_parameter("feature_names", features_names)
# output.add_data([negative_path], length=len(negative_cases), positive=False)
# output.add_data([positive_paths[i][10:] for i in positive_indices], length=total_len, positive=True)
#
# bad_negative = 0
# total_negative = 0
# for file in negative_cases:
#     ecg = io.matlab.loadmat(negative_path + "/" + file)["val"][0]
#     try:
#         features, features_names = FeatureExtraction.generate_features(ecg, hz)
#         features.append(0)
#         features_names.append("Abnormal")
#         X.append(features)
#         y.append(0)
#     except BadFeatureGenerationException:
#         print(f"0: {file} [negative]")
#         bad_negative += 1
#     total_negative += 1
#
# print(f"Bad: {bad_negative}/{total_negative}")
# print(f"Good: {bad_positive}/{total_positive}")


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

make_cool_roc_curve(y_tests, probas, title="Basis")

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
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

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

make_cool_roc_curve(y_tests, probas, True, "Best")

output.fill_binary_metrics(matrix_in_list(y_tests), matrix_in_list(y_preds), path="best")

end = time.time()
output.add_performance("total_time", end - start)
output.generate_json("%s.json" % datetime.datetime.now().strftime("%y%m%d-%H%M"))
