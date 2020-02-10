#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot
from data import DATA_LIST


class AdaBoostDT(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "boosting"
        super().__init__()

    def construct_model(self, params, **kwargs):
        pipe = Pipeline(
            [
                (
                    "model",
                    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**kwargs)),
                )
            ]
        )
        self.model = GridSearchCV(
            pipe, params, return_train_score=True, cv=self.cv, scoring=self.scoring
        )


for dat in DATA_LIST:
    boost = AdaBoostDT()

    if dat.data_name == "credit":
        ccp_alpha = 0.006
        params = {"model__n_estimators": range(10, 121, 10)}
    elif dat.data_name == "wine":
        ccp_alpha = 0.002
        params = {"model__n_estimators": [10, 30, 50, 75, 100, 200, 300, 500]}

    boost.construct_model(params=params, ccp_alpha=ccp_alpha)
    res_df = boost.run_experiment(dat)

    mean_df = res_df.groupby("param_model__n_estimators").mean()
    std_df = res_df.groupby("param_model__n_estimators").std()

    generate_plot(
        mean_df=mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_boosting_acc.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"Decision tree with Boosting accuracy on {dat.data_name} data",
    )
    
    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="runtime",
        fname=f"{dat.data_name}_boosting_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
        title=f"Decision tree with Boosting runtime on {dat.data_name} data",
    )
