import datetime
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold


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

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

train_start = time.time()
clf = RandomForestClassifier(bootstrap=False, n_estimators=1400, max_features="sqrt", max_depth=50, n_jobs=-1, random_state=RANDOM_STATE)

output.add_model_parameter("param", clf.get_params(deep=True))

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

make_cool_roc_curve(y_tests, probas, title="ECG")
print(f"Accuracy model: {accuracy_score(matrix_in_list(y_tests), matrix_in_list(y_preds))}")

prec, recall, fscore, _ = precision_recall_fscore_support(matrix_in_list(y_tests), matrix_in_list(y_preds), average="binary")
print(f"Precision: {prec} | Recall: {recall} | F Score: {fscore}")

output.fill_binary_metrics(matrix_in_list(y_tests), matrix_in_list(y_preds), path="ecg")

end = time.time()
output.add_performance("total_time", end - start)
output.generate_json("%s.json" % datetime.datetime.now().strftime("%y%m%d-%H%M"))
