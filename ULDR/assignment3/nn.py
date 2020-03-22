import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from load_data import DATA

nn = MLPClassifier(
    hidden_layer_sizes=(144,),
    activation="relu",
    max_iter=1000,
    early_stopping=True,
    learning_rate_init=0.01,
)

# raw data
print("Running NN on raw data...")
X, y = DATA["credit"]
cv_results = cross_validate(
    nn,
    X,
    y,
    scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"],
    cv=5,
    n_jobs=-1,
    return_train_score=True,
)
res_mean = {key: [] for key in cv_results}
res_std = {key: [] for key in cv_results}
for key in cv_results:
    res_mean[key].append(cv_results[key].mean())
    res_std[key].append(cv_results[key].std())

ind = ["raw"]

with open("data/dim_red_data.pkl", "rb") as f:
    data_red = pickle.load(f)["credit"]

# dimension reduction data
print("Running NN on dimension reduction data...")
for algo_key in data_red:
    ind.append(algo_key)
    X, y = data_red[algo_key]
    cv_results = cross_validate(
        nn,
        X,
        y,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )
    for key in cv_results:
        res_mean[key].append(cv_results[key].mean())
        res_std[key].append(cv_results[key].std())

# cluster data
print("Running NN on cluster data...")
for cluster in ["kmeans", "EM"]:
    for algo_key in data_red:
        ind.append(f"{cluster}_{algo_key}")
        X = pd.read_csv(f"data/{cluster}_labels_{algo_key}_credit.csv")
        cv_results = cross_validate(
            nn,
            X,
            y,
            scoring=[
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ],
            cv=5,
            n_jobs=-1,
            return_train_score=True,
        )
        for key in cv_results:
            res_mean[key].append(cv_results[key].mean())
            res_std[key].append(cv_results[key].std())

df_mean = pd.DataFrame(res_mean, index=ind)
df_std = pd.DataFrame(res_std, index=ind)

for i in range(int(len(df_mean.columns) / 2)):
    cols = df_mean.columns[i * 2 : i * 2 + 2]
    fname = f"output/nn_{'_'.join(cols[0].split('_')[1:])}.png"
    df_mean[cols].plot(yerr=df_std[cols], rot=90, xticks=range(len(ind)))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
