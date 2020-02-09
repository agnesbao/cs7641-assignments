# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from best_models import BEST_MODELS
from data import DATA_LIST


def run_training_size(clf, data):
    res = {
        "training_perc": [],
        "training_size": [],
        "training_score": [],
        "test_score": [],
    }
    for i in np.linspace(0.1, 0.9, 9):
        res["training_perc"].append(i)
        X_train_subset, _, y_train_subset, _ = train_test_split(
            data.X_train, data.y_train, train_size=i
        )
        res["training_size"].append(X_train_subset.shape[0])
        clf.fit(X_train_subset, y_train_subset)
        res["training_score"].append(clf.score(X_train_subset, y_train_subset))
        res["test_score"].append(clf.score(data.X_test, data.y_test))
    res["training_perc"].append(1)
    res["training_size"].append(data.X_train.shape[0])
    clf.fit(data.X_train, data.y_train)
    res["training_score"].append(clf.score(data.X_train, data.y_train))
    res["test_score"].append(clf.score(data.X_test, data.y_test))

    df = pd.DataFrame(res)

    return df


for dat in DATA_LIST:
    for model in BEST_MODELS:
        clf = Pipeline(
            [("scaler", StandardScaler()), ("model", BEST_MODELS[model][dat.data_name])]
        )

        df_list = []
        for n in range(20):
            df_list.append(run_training_size(clf, dat))

        res_df = pd.concat(df_list)
        mean_df = res_df.groupby("training_size").mean()
        std_df = res_df.groupby("training_size").std()
        mean_df.plot(
            y=["training_score", "test_score"],
            yerr=std_df[["training_score", "test_score"]],
            title=f"{model} accuracy vs training size on {dat.data_name} data",
        )
        plt.ylabel("accuracy")
        plt.savefig(
            os.path.join("output", f"{dat.data_name}_{model}_training_size.png")
        )
        plt.clf()


#%% max_iter
MLP_MODEL = {
    "credit": MLPClassifier(
        hidden_layer_sizes=(144,), activation="relu", learning_rate_init=0.0001,
    ),
    "wine": MLPClassifier(
        hidden_layer_sizes=(33,), activation="relu", learning_rate_init=0.0001,
    ),
}
for dat in DATA_LIST:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", MLP_MODEL[dat.data_name])])
    pipe.fit(dat.X_train, dat.y_train)
    n_iter = pipe.steps[1][1].n_iter_
    params = {"model__max_iter": np.linspace(10, n_iter, 10).astype(int)}
    clf = GridSearchCV(pipe, params, return_train_score=True, cv=10)
    clf.fit(dat.X_train, dat.y_train)
    res_df = pd.DataFrame(clf.cv_results_)
    res_df = res_df.set_index("param_model__max_iter")
    res_df.plot(
        y=["mean_train_score", "mean_test_score"],
        yerr=res_df[["std_train_score", "std_test_score"]].rename(
            columns={
                "std_train_score": "mean_train_score",
                "std_test_score": "mean_test_score",
            }
        ),
        title=f"MLP accuracy vs max_iter on {dat.data_name} data",
    )
    plt.ylabel("accuracy")
    plt.savefig(os.path.join("output", f"{dat.data_name}_mlp_max_iter.png"))
    plt.clf()


#%% Final performance
for dat in DATA_LIST:
    final_score = {"model": [], "training_score": [], "test_score": []}
    for model in BEST_MODELS:
        clf = Pipeline(
            [("scaler", StandardScaler()), ("model", BEST_MODELS[model][dat.data_name])]
        )
        clf.fit(dat.X_train, dat.y_train)
        final_score["model"].append(model)
        final_score["training_score"].append(clf.score(dat.X_train, dat.y_train))
        final_score["test_score"].append(clf.score(dat.X_test, dat.y_test))

    final_score_df = pd.DataFrame(final_score).sort_values(by="test_score")
    final_score_df.plot(
        x="model",
        y=["training_score", "test_score"],
        style=".-",
        title=f"Final accuracy on test data vs model on {dat.data_name} data",
    )
    plt.savefig(os.path.join("output", f"{dat.data_name}_final_scores.png"))
    plt.clf()
