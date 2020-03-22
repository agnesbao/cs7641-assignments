import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from examine_cluster import examine_credit_cluster

with open("data/dim_red_data.pkl", "rb") as f:
    DATA = pickle.load(f)

RUN_DATA = ["credit", "fashion"]
NUM_CLUSTERS = range(2, 20)


def run_kmeans(X, y):
    iner = []
    ami = []
    sc = []
    k_labels = {}
    for k in NUM_CLUSTERS:
        print(f"  Running for {k} clusters...")
        model = KMeans(n_clusters=k, random_state=1, n_jobs=-1)
        model.fit(X)
        iner.append(model.inertia_)
        ami.append(adjusted_mutual_info_score(y, model.labels_))
        sc.append(silhouette_score(X, model.labels_))
        k_labels[k] = model.labels_
    res_df = pd.DataFrame(
        data={
            "Adjusted Mutual Info": ami,
            "Silhouette Coefficient": sc,
            "KMeans Inertia": iner,
        },
        index=NUM_CLUSTERS,
    )
    k_labels_df = pd.DataFrame(data=k_labels)
    return res_df, k_labels_df


for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running KMeans on {data_key} data")
    for algo_key in DATA[data_key]:
        print(f"..Running on {algo_key} transformed data")
        X, y = DATA[data_key][algo_key]
        res_df, k_labels_df = run_kmeans(X, y)
        k_labels_df.to_csv(f"data/kmeans_labels_{algo_key}_{data_key}.csv", index=False)
        res_df.plot(
            subplots=True,
            style=".-",
            title=f"KMeans performance vs n_clusters on {algo_key} transformed {data_key} data",
        )
        plt.xlabel("n_clusters")
        plt.savefig(f"output/kmeans_{algo_key}_{data_key}.png")
        plt.close()

        examine_credit_cluster(
            X[:, :2],
            y,
            title=f"True label of {algo_key} transformed {data_key} data",
            fname=f"output/true_cluster_{algo_key}_{data_key}.png",
        )

        examine_credit_cluster(
            X[:, :2],
            k_labels_df[len(np.unique(y))],
            title=f"KMeans on {algo_key} transformed {data_key} data",
            fname=f"output/kmeans_cluster_{algo_key}_{data_key}.png",
        )
