# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:40:45 2020

@author: Xiaojun
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot


class SVM(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "svm"

    def construct_model(self, scale: bool):
        if scale:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", SVC())])
        else:
            pipe = Pipeline([("model", SVC())])
        params = {
            "model__kernel": ["linear", "poly", "rbf"],
            "model__degree": [2,3],
        }
        self.model = GridSearchCV(pipe, params, return_train_score=True, n_jobs=-1)


for dat in data_list[0]:
    svm = SVM()
    svm.construct_model(scale=False)
    res_df = svm.run_experiment(dat, n_iter=1)

    mean_df = res_df.groupby("param_model__kernel").mean()
    std_df = res_df.groupby("param_model__kernel").std()

    generate_plot(
        mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        title="SVM model performance",
        fname=dat.data_name + "_svm_acc.png",
    )

    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="accuracy",
        title="SVM model runtime",
        fname=dat.data_name + "_svm_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
    )

    # with standard scaling
    svm_scale = SVM()
    svm_scale.construct_model(scale=True)
    svm_scale.run_experiment(dat)

    mean_df = res_df.groupby("param_model__n_neighbors").mean()
    std_df = res_df.groupby("param_model__n_neighbors").std()

    generate_plot(
        mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        title="KNN model performance with standard scaling",
        fname=dat.data_name + "_svm_acc_scale.png",
    )
