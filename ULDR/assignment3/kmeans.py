import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import TOP_FEATURES
from examine_cluster import examine_credit_cluster
from examine_cluster import plot_fashion_cluster

RUN_DATA = ["credit"]
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
    X, y = DATA[data_key]
    res_df, k_labels_df = run_kmeans(X, y)
    k_labels_df.to_csv(f"data/kmeans_labels_{data_key}.csv", index=False)
    res_df.plot(
        subplots=True,
        style=".-",
        title=f"KMeans performance vs n_clusters on {data_key} data",
    )
    plt.xlabel("n_clusters")
    plt.savefig(f"output/kmeans_{data_key}.png")
    plt.close()

    if data_key == "credit":
        examine_credit_cluster(
            X.values[:, TOP_FEATURES[:2]],
            y,
            title="True Label",
            xylabel=DATA["credit"][0].columns[TOP_FEATURES[:2]],
            fname="output/true_cluster_credit.png",
        )
        examine_credit_cluster(
            X.values[:, TOP_FEATURES[:2]],
            k_labels_df[y.nunique()],
            title="KMeans",
            xylabel=DATA["credit"][0].columns[TOP_FEATURES[:2]],
            fname="output/kmeans_cluster_credit.png",
        )

    if data_key == "fashion":
        plot_fashion_cluster(
            X, k_labels_df[len(np.unique(y))], fname="output/kmeans_cluster_fashion.png"
        )
