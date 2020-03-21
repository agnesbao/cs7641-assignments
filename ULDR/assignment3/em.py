import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import WINE_TOP_FEATURES
from examine_cluster import examine_wine_cluster
from examine_cluster import plot_fashion_cluster


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
    print(f"Running EM on {data_key} data")
    X, y = DATA[data_key]
    if data_key == "wine":
        X = StandardScaler().fit_transform(X)
    res_df, k_labels_df = run_em(X, y)
    k_labels_df.to_csv(f"data/EM_labels_{data_key}.csv", index=False)
    res_df.plot(
        subplots=True,
        style=".-",
        title=f"EM performance vs n_clusters on {data_key} data",
    )
    plt.xlabel("n_clusters")
    plt.savefig(f"output/EM_{data_key}.png")
    plt.close()

    if data_key == "wine":
        examine_wine_cluster(
            X[:, WINE_TOP_FEATURES[:2]],
            k_labels_df[y.nunique()],
            title="EM",
            xylabel=DATA["wine"][0].columns[WINE_TOP_FEATURES[:2]],
            fname="output/EM_cluster_wine.png",
        )

    if data_key == "fashion":
        plot_fashion_cluster(
            X, k_labels_df[len(np.unique(y))], fname="output/EM_cluster_fashion.png"
        )
