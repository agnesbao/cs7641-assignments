# -*- coding: utf-8 -*-
"""
@author: Xiaojun
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from AbstractModelClass import _AbstractModelClass
from AbstractModelClass import generate_plot
from data import DATA_LIST


class MLP(_AbstractModelClass):
    def __init__(self):
        self.algo_name = "mlp"
        super().__init__()

    def construct_model(self, params):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        max_iter=1000, early_stopping=True, learning_rate_init=0.01
                    ),
                ),
            ]
        )
        self.model = GridSearchCV(
            pipe, params, return_train_score=True, cv=self.cv, scoring=self.scoring
        )


for dat in DATA_LIST:
    mlp = MLP()

    params = {
        "model__hidden_layer_sizes": [
            (dat.n_features,),
            (dat.n_features * 2,),
            (dat.n_features * 3,),
            (dat.n_features * 2, dat.n_features),
            (dat.n_features * 2, dat.n_features * 2),
        ],
        "model__activation": ["tanh", "relu"],
    }
    mlp.construct_model(params)
    res_df = mlp.run_experiment(dat)

    mean_df = res_df.groupby(
        ["param_model__activation", "param_model__hidden_layer_sizes"], sort=False
    ).mean()
    std_df = res_df.groupby(
        ["param_model__activation", "param_model__hidden_layer_sizes"], sort=False
    ).std()

    generate_plot(
        mean_df[["mean_train_score", "mean_test_score"]],
        ylabel="accuracy",
        fname=f"{dat.data_name}_mlp_acc.png",
        std_df=std_df[["mean_train_score", "mean_test_score"]],
        title=f"MLP accuracy on {dat.data_name} data",
        rot=45,
        figsize=(7, 7),
    )

    generate_plot(
        mean_df[["mean_fit_time", "mean_score_time"]],
        ylabel="runtime",
        fname=f"{dat.data_name}_mlp_runtime.png",
        std_df=std_df[["mean_fit_time", "mean_score_time"]],
        title=f"MLP runtime on {dat.data_name} data",
        rot=45,
        figsize=(7, 7),
    )
