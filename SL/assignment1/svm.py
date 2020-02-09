# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot
from data import DATA_LIST


class SVM(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "svm"
        super().__init__()

    def construct_model(self, params):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", SVC())])
        self.model = GridSearchCV(
            pipe, params, return_train_score=True, cv=self.cv, scoring=self.scoring
        )


for dat in DATA_LIST:
    svm = SVM()

    params = {"model__kernel": ["linear", "rbf"]}
    svm.construct_model(params)
    res_df1 = svm.run_experiment(dat)

    params = {"model__kernel": ["poly"], "model__degree": range(2, 7)}
    svm.construct_model(params)
    res_df2 = svm.run_experiment(dat)

    res_df = pd.concat([res_df1, res_df2])
    res_df.to_csv(os.path.join("data", f"{dat.data_name}_{svm.algo_name}.csv"))
    res_df["param_model__degree"] = res_df["param_model__degree"].astype(str)

    mean_df = res_df.groupby(["param_model__kernel", "param_model__degree"]).mean()
    std_df = res_df.groupby(["param_model__kernel", "param_model__degree"]).std()

    generate_plot(
        mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_svm_acc.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"SVM accuracy on {dat.data_name} data",
    )

    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="runtime",
        fname=f"{dat.data_name}_svm_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
        title=f"SVM runtime on {dat.data_name} data",
    )
