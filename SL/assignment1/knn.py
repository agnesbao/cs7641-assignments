# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:27:54 2020

@author: Xiaojun
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def construct_knn(scale: bool):
    if scale:
        pipe = Pipeline(
                [("scaler", StandardScaler()),
                 ("model", KNeighborsClassifier(weights="distance"))]
                )
    else:
        pipe = Pipeline(
                [("model", KNeighborsClassifier(weights="distance"))]
                )
    params = {"model__n_neighbors": range(1,50,3)}
    clf = GridSearchCV(pipe, params, return_train_score=True)
    
    return clf

def run_experiment(data, clf, n_iter = 20):
    df_list = []
    for n in range(n_iter):
        clf.fit(data.X, data.y)
        df_list.append(pd.DataFrame(clf.cv_results_))
    res_df = pd.concat(df_list)
    
    return res_df

def generate_plot(mean_df, ylabel, title, fname, std_df=None):
    if std_df is not None:
        mean_df.plot(yerr=std_df, 
                     title=title)
    else:
        mean_df.plot(style=".-",
                     title=title)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("output", fname))
    plt.clf()        

clf = construct_knn(scale=False)
res_df = run_experiment(dat, clf)

mean_df = res_df.groupby("param_model__n_neighbors").mean()
std_df = res_df.groupby("param_model__n_neighbors").std()

generate_plot(mean_df[["mean_train_score", "mean_test_score"]],
              ylabel = "accuracy",
              title="KNN model performance",
              fname=dat.data_name+"_knn_acc.png")

generate_plot(mean_df[["mean_fit_time", "mean_score_time"]],
              ylabel = "accuracy",
              title="KNN model runtime",
              fname=dat.data_name+"_knn_runtime.png",
              std_df=std_df[["mean_fit_time", "mean_score_time"]])

# with standard scaling
clf = construct_knn(scale=True)
res_df = run_experiment(dat, clf)

mean_df = res_df.groupby("param_model__n_neighbors").mean()
std_df = res_df.groupby("param_model__n_neighbors").std()

generate_plot(mean_df[["mean_train_score", "mean_test_score"]],
              ylabel = "accuracy",
              title="KNN model performance with standard scaling",
              fname=dat.data_name+"_knn_acc_scale.png")