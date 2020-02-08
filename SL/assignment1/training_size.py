# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:26:47 2020

@author: Xiaojun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


def run_training_size(clf, data):
    res = {
        "training_perc": [],
        "training_size": [],
        "training_score": [],
        "test_score": [],
    }
    for i in np.arange(0.1, 1, 0.1):
        res["training_perc"].append(i)
        X_train, _, y_train, _ = train_test_split(
            data.X_train, data.y_train, train_size=i
        )
        res["training_size"].append(X_train.shape[0])
        clf.fit(X_train, y_train)
        res["training_score"].append(clf.score(X_train, y_train))
        res["test_score"].append(clf.score(data.X_test, data.y_test))
    res["training_perc"].append(1)
    res["training_size"].append(data.X_train.shape[0])
    clf.fit(data.X_train, data.y_train)
    res["training_score"].append(clf.score(data.X_train, data.y_train))
    res["test_score"].append(clf.score(data.X_test, data.y_test))

    res_df = pd.DataFrame(res)

    return res_df


clf = KNeighborsClassifier(weights="distance")

df_list = []
for n in range(20):
    df_list.append(run_training_size(clf, dat))

res_df = pd.concat(df_list)
mean_df = res_df.groupby("training_size").mean()
std_df = res_df.groupby("training_size").std()
mean_df.plot(
    y=["training_score", "test_score"], yerr=std_df[["training_score", "test_score"]]
)
