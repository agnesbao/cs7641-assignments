# -*- coding: utf-8 -*-
"""
@author: Xiaojun

Prepare data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, data_name):
        self.data_name = data_name

    def load_data(self):
        if self.data_name == "credit":
            # https://www.openml.org/d/31
            dat = pd.read_csv(os.path.join("data", "dataset_31_credit-g.csv"))
        elif self.data_name == "wine":
            # http://archive.ics.uci.edu/ml/datasets/Wine+Quality
            dat = pd.read_csv(os.path.join("data", "winequality-white.csv"), sep=";")
        self.X = dat.iloc[:, :-1]
        self.y = dat.iloc[:, -1]

        return self.X, self.y

    def preprocess_data(self):
        X_cat = self.X.select_dtypes(object)
        if not X_cat.empty:
            X_cat = pd.get_dummies(X_cat, drop_first=True)
            self.X = self.X.select_dtypes("number").join(X_cat)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )
        self.n_class = self.y.nunique()
        self.n_features = self.X.shape[1]

    def plot_X(self):
        self.X.plot.box(
            logy=True, rot=90, figsize=(7, 7), title="Distribution of features"
        )
        plt.savefig(os.path.join("output", self.data_name + "_X.png"))
        plt.clf()

    def plot_y(self):
        self.y.value_counts().plot.pie(figsize=(5, 5), title="Distribution of classes")
        plt.savefig(os.path.join("output", self.data_name + "_y.png"))
        plt.clf()


DATA_LIST = []
for dname in ["credit", "wine"]:
    dat = Data(dname)
    dat.load_data()
    dat.preprocess_data()
    dat.plot_X()
    dat.plot_y()
    DATA_LIST.append(dat)
