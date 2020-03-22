import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from examine_cluster import examine_credit_cluster

with open("data/dim_red_data.pkl", "rb") as f:
    DATA = pickle.load(f)

RUN_DATA = ["credit", "fashion"]
NUM_CLUSTERS = range(2, 20)


def run_em(X, y):
    ami = []
    sc = []
    bic = []
    k_labels = {}
    for k in NUM_CLUSTERS:
        print(f"  Running for {k} clusters...")
        model = GaussianMixture(n_components=k, random_state=1)
        y_pred = model.fit_predict(X)
        ami.append(adjusted_mutual_info_score(y, y_pred))
        sc.append(silhouette_score(X, y_pred))
        bic.append((model.bic(X)))
        k_labels[k] = y_pred
    res_df = pd.DataFrame(
        data={
            "Adjusted Mutual Info": ami,
            "Silhouette Coefficient": sc,
            "Bayesian information criterion": bic,
        },
        index=NUM_CLUSTERS,
    )
    k_labels_df = pd.DataFrame(data=k_labels)
    return res_df, k_labels_df


for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running EM on {data_key} data")
    for algo_key in DATA[data_key]:
        print(f"..Running on {algo_key} transformed data")
        X, y = DATA[data_key][algo_key]
        res_df, k_labels_df = run_em(X, y)
        k_labels_df.to_csv(f"data/EM_labels_{algo_key}_{data_key}.csv", index=False)
        res_df.plot(
            subplots=True,
            style=".-",
            title=f"EM performance vs n_clusters on {algo_key} transformed {data_key} data",
        )
        plt.xlabel("n_clusters")
        plt.savefig(f"output/EM_{algo_key}_{data_key}.png")
        plt.close()

        examine_credit_cluster(
            X[:, :2],
            k_labels_df[len(np.unique(y))],
            title=f"EM on {algo_key} transformed {data_key} data",
            fname=f"output/EM_cluster_{algo_key}_{data_key}.png",
        )
