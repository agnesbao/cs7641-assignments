# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot
from data import DATA_LIST


class KNN(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "knn"
        super().__init__()

    def construct_model(self, scale: bool):
        if scale:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(weights="distance")),
                ]
            )
        else:
            pipe = Pipeline([("model", KNeighborsClassifier(weights="distance"))])
        params = {"model__n_neighbors": range(1, 50, 4)}
        self.model = GridSearchCV(
            pipe, params, return_train_score=True, cv=self.cv, scoring=self.scoring
        )


for dat in DATA_LIST:
    knn = KNN()
    knn.construct_model(scale=False)
    res_df = knn.run_experiment(dat)

    mean_df = res_df.groupby("param_model__n_neighbors").mean()
    std_df = res_df.groupby("param_model__n_neighbors").std()

    generate_plot(
        mean_df=mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_knn_acc.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"KNN accuracy on {dat.data_name} data",
    )

    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="runtime",
        fname=f"{dat.data_name}_knn_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
        title=f"KNN runtime on {dat.data_name} data",
    )

    # with standard scaling
    knn_scale = KNN()
    knn_scale.construct_model(scale=True)
    res_df = knn_scale.run_experiment(dat)

    mean_df = res_df.groupby("param_model__n_neighbors").mean()
    std_df = res_df.groupby("param_model__n_neighbors").std()

    generate_plot(
        mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_knn_acc_scale.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"KNN accuracy with standard scaling on {dat.data_name} data",
    )
