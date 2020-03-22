import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import examine_credit_cluster
from examine_cluster import plot_fashion_cluster

RUN_DATA = ["credit", "fashion"]

for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running RCA on {data_key} data")
    X, y = DATA[data_key]
    rca = GaussianRandomProjection(n_components=X.shape[1], random_state=0)
    X_rca = rca.fit_transform(X)

    # RC
    examine_credit_cluster(
        X_rca[:, :2],
        y,
        title=f"RCA transformation on {data_key} data",
        xylabel=["RC1", "RC2"],
        fname=f"output/rca_rc_cluster_{data_key}.png",
    )

    if data_key == "fashion":
        # recon vs k
        sample = []
        for nc in np.linspace(2, X.shape[1], 25).astype(int):
            print(f"Reconstructing with {nc} RC...")
            rca = GaussianRandomProjection(n_components=nc, random_state=0)
            X_rca = rca.fit_transform(X)
            X_recon = np.dot(X_rca, np.linalg.pinv(rca.components_.T))
            sample.append(X_recon[0])
        plot_fashion_cluster(
            np.array(sample), range(25), fname="output/rca_recon_vs_k_fashion.png"
        )
    elif data_key == "credit":
        corr_mean = []
        corr_std = []
        for nc in range(1, X.shape[1]):
            print(f"Reconstructing with {nc} RC...")
            corr = []
            for i in range(100):
                rca = GaussianRandomProjection(n_components=nc)
                X_rca = rca.fit_transform(X)
                X_recon = np.dot(X_rca, np.linalg.pinv(rca.components_.T))
                corr.append(np.corrcoef(X.values.flatten(), X_recon.flatten())[0, 1])
            corr_mean.append(np.mean(corr))
            corr_std.append(np.std(corr))
        corr_mean = np.array(corr_mean)
        corr_std = np.array(corr_std)
        plt.plot(range(1, X.shape[1]), corr_mean)
        plt.fill_between(
            range(1, X.shape[1]), corr_mean - corr_std, corr_mean + corr_std, alpha=0.3
        )
        plt.xlabel("n_components")
        plt.ylabel("corr between raw and recomposed data")
        plt.title(f"Data reconstruction quality vs RCA n_components on {data_key} data")
        plt.savefig(f"output/rca_recon_vs_k_{data_key}.png")
        plt.close()
