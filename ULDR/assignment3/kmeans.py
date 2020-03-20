import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from load_data import DATA
import matplotlib.pyplot as plt

NUM_CLUSTERS = range(2, 16)


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
            "KMeans Inertia": iner,
            "Silhouette Coefficient": sc,
        },
        index=NUM_CLUSTERS,
    )
    k_labels_df = pd.DataFrame(data=k_labels)
    return res_df, k_labels_df


for data_key in DATA:
    print(f"Running KMeans on {data_key} data")
    X, y = DATA[data_key]
    if data_key == "wine":
        X = StandardScaler().fit_transform(X)
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
