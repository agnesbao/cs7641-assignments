import numpy as np
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

    # IC
    examine_credit_cluster(
        X_ica[:, top_kurt_ind[:2]],
        y,
        title=f"ICA transformation on {data_key} data",
        xylabel=["IC1", "IC2"],
        fname=f"output/ica_ic_cluster_{data_key}.png",
    )

    if data_key == "fashion":
        # IC
        plot_fashion_cluster(
            ica.components_[top_kurt_ind[:25], :],
            range(25),
            fname="output/ica_ic_fashion.png",
        )

        # reconstructed image
        X_recon = ica.inverse_transform(X_ica)
        plot_fashion_cluster(X_recon, y, fname="output/ica_reconstructed_fashion.png")

        # recon vs k
        sample = []
        for nc in range(2, 201, 8):
            print(f"Reconstructing with {nc} PC...")
            ica = FastICA(n_components=nc, whiten=True, random_state=0)
            X_recon = ica.inverse_transform(ica.fit_transform(X))
            sample.append(X_recon[0])
        plot_fashion_cluster(
            np.array(sample), range(25), fname="output/ica_recon_vs_k_fashion.png"
        )

    elif data_key == "credit":
        # reconstruct data
        corr_mean = []
        for nc in range(1, X.shape[1]):
            print(f"Reconstructing with {nc} IC...")
            ica = FastICA(n_components=nc, whiten=True, random_state=0)
            X_recon = ica.inverse_transform(ica.fit_transform(X))
            corr = np.corrcoef(X.values.flatten(), X_recon.flatten())[0, 1]
            corr_mean.append(corr)
        plt.plot(range(1, X.shape[1]), corr_mean)
        plt.xlabel("n_components")
        plt.ylabel("corr between raw and recomposed data")
        plt.title(f"Data reconstruction quality vs ICA n_components on {data_key} data")
        plt.savefig(f"output/ica_recon_vs_k_{data_key}.png")
        plt.close()
