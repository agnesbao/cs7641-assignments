from sklearn.manifold import TSNE

from load_data import DATA
from examine_cluster import examine_credit_cluster

RUN_DATA = ["fashion"]

for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running TSNE on {data_key} data")
    X, y = DATA[data_key]

    tsne = TSNE(n_components=3, random_state=0, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    examine_credit_cluster(
        X_tsne[:, :2],
        y,
        title=f"TSNE on {data_key} data",
        xylabel=["", ""],
        fname=f"output/tsne_cluster_{data_key}.png",
    )
