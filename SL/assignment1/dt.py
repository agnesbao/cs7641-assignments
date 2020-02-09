# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot
from data import DATA_LIST


class DT(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "dt"
        super().__init__()

    def get_cpp_alphas(self, data):
        clf = DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(data.X_train, data.y_train)
        path_df = pd.DataFrame(path)
        path_df[:-1].plot(x="ccp_alphas", y="impurities", style=".-", legend=False)
        plt.ylabel("total impurity of leaves")
        plt.savefig(os.path.join("output", f"{data.data_name}_ccp_alphas.png"))
        plt.clf()
        self.ccp_alphas = np.linspace(
            0, path_df[path_df["impurities"] < 0.5]["ccp_alphas"].iloc[-2], 100
        )

        res_dict = {
            "cpp_alphas": [],
            "node_count": [],
            "max_depth": [],
            "n_leaves": [],
        }
        for a in self.ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=a)
            clf.fit(data.X_train, data.y_train)
            res_dict["cpp_alphas"].append(a)
            res_dict["node_count"].append(clf.tree_.node_count)
            res_dict["max_depth"].append(clf.tree_.max_depth)
            res_dict["n_leaves"].append(clf.tree_.n_leaves)
        res_df = pd.DataFrame(res_dict)
        res_df = res_df.set_index("cpp_alphas")
        res_df.plot(
            subplots=True,
            style=".-",
            figsize=(10, 5),
            title="Decision tree complexity vs ccp_alpha",
        )
        plt.savefig(os.path.join("output", f"{data.data_name}_ccp_alphas_tree.png"))
        plt.clf()

    def construct_model(self, scale: bool):
        if scale:
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("model", DecisionTreeClassifier()),]
            )
        else:
            pipe = Pipeline([("model", DecisionTreeClassifier())])

        params = {"model__ccp_alpha": self.ccp_alphas}
        self.model = GridSearchCV(
            pipe, params, return_train_score=True, cv=self.cv, scoring=self.scoring
        )


for dat in DATA_LIST:
    dt = DT()
    dt.get_cpp_alphas(dat)

    #%%
    dt.construct_model(scale=False)

    res_df = dt.run_experiment(dat)

    mean_df = res_df.groupby("param_model__ccp_alpha").mean()
    std_df = res_df.groupby("param_model__ccp_alpha").std()

    generate_plot(
        mean_df=mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_dt_acc.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"DT accuracy on {dat.data_name} data",
    )

    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="runtime",
        fname=f"{dat.data_name}_dt_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
        title=f"DT runtime on {dat.data_name} data",
    )

    if dat.data_name == "credit":
        # with standard scaling
        dt_scale = DT()
        dt_scale.get_cpp_alphas(dat)
        dt_scale.construct_model(scale=True)
        res_df = dt_scale.run_experiment(dat)

        mean_df = res_df.groupby("param_model__ccp_alpha").mean()
        std_df = res_df.groupby("param_model__ccp_alpha").std()

        generate_plot(
            mean_df=mean_df[["mean_train_score", "mean_test_score"]],
            ylabel="accuracy",
            fname=f"{dat.data_name}_dt_acc_scale.png",
            std_df=std_df[["mean_train_score", "mean_test_score"]],
            title=f"DT accuracy with standard scaling on {dat.data_name} data",
        )
