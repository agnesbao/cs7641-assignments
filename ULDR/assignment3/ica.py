import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import examine_credit_cluster
from examine_cluster import plot_fashion_cluster

RUN_DATA = ["credit"]

for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running ICA on {data_key} data")
    X, y = DATA[data_key]
    if data_key == "credit":
        ica = FastICA(whiten=True, random_state=0)
    elif data_key == "fashion":
        ica = FastICA(n_components=200, whiten=True, random_state=0)
    X_ica = ica.fit_transform(X)
    kurt = kurtosis(X_ica)
    plt.plot(kurt)
    plt.xlabel("IC")
    plt.ylabel("kurtosis")
    plt.title(f"ICA kurtosis on {data_key} data")
    plt.savefig(f"output/ica_kurtosis_{data_key}.png")
    plt.close()

    top_kurt_ind = (-kurt).argsort()

    if data_key == "fashion":
        plot_fashion_cluster(
            ica.components_[top_kurt_ind[:25], :],
            range(25),
            fname="output/ica_ic_fashion.png",
        )
        # reconstructed image
        X_recon = ica.inverse_transform(X_ica)
        plot_fashion_cluster(X_recon, y, fname="output/ica_reconstructed_fashion.png")
    elif data_key == "credit":
        examine_credit_cluster(
            X_ica[:, top_kurt_ind[:2]],
            y,
            title="IC1 and IC2 on credit data",
            xylabel=["IC1", "IC2"],
            fname="output/ica_ic_credit.png",
        )
        # reconstruct data
        corr_mean = []
        corr_std = []
        for nc in range(1, X.shape[1]):
            print(f"Reconstructing with {nc} PC...")
            ica = FastICA(n_components=nc, whiten=True, random_state=0)
            X_recon = ica.inverse_transform(ica.fit_transform(X))
            corr = X.corrwith(pd.DataFrame(X_recon, columns=X.columns), axis=1)
            corr_mean.append(corr.mean())
            corr_std.append(corr.std())
        corr_mean = np.array(corr_mean)
        corr_std = np.array(corr_std)
        plt.plot(range(1, X.shape[1]), corr_mean)
        plt.fill_between(
            range(1, X.shape[1]), corr_mean - corr_std, corr_mean + corr_std, alpha=0.3
        )
        plt.xlabel("n_components")
        plt.ylabel("corr between raw and recomposed data")
        plt.title(f"Data reconstruction quality vs ICA n_components on {data_key} data")
        plt.savefig(f"output/ica_recon_vs_k_{data_key}.png")
        plt.close()
