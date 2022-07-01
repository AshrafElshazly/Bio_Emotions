from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import time


def preparing_ML_data(path):
    data = pd.read_csv(path)
    group_kfold = GroupKFold(n_splits=14)
    X = np.array(data.loc[:, 'ECG_Rate_Mean':'HRV_SampEn'])
    y = np.array(data['stress_bin'])
    groups = np.array(data['participant'])
    for train, test in group_kfold.split(X, y, groups):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

    return X, y, groups, X_train, X_test, y_train, y_test


def run_clf(clf, X, y, groups, X_test, y_test):
    cv = GroupKFold(n_splits=10)
    score = []
    runtime = []
    for fold, (train, test) in enumerate(cv.split(X, y, groups)):
        clf.fit(X[train], y[train])
        start = time.time()
        score.append(clf.score(X_test, y_test))
        runtime.append(time.time() - start)

    return score, runtime


def supervised_model(train_data):
    results = []
    X, y, groups, X_train, X_test, y_train, y_test = preparing_ML_data(
        train_data)
    model = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVC(gamma=2, C=1))
    score, runtime = run_clf(model, X, y, groups, X_test, y_test)
    results.append(["SVC", round(np.mean(score)*100, 1),
                   round(np.mean(runtime), 9)])
    result = pd.DataFrame(results, columns=['Name', 'Accuracy', 'Runtime'])

    return model, result
