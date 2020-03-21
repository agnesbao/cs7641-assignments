import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import examine_credit_cluster
from examine_cluster import plot_fashion_cluster

for data_key in DATA:
    print(f"Running RCA on {data_key} data")
    X, y = DATA[data_key]
    rca = GaussianRandomProjection(n_components=200, whiten=True, random_state=0)
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
