import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import examine_credit_cluster
from examine_cluster import plot_fashion_cluster

RUN_DATA = ["credit"]

for data_key in DATA:
    if data_key not in RUN_DATA:
        continue
    print(f"Running PCA on {data_key} data")
    X, y = DATA[data_key]
    # center data
    X = X - X.mean()

    pca = PCA(whiten=True, random_state=0)
    X_pca = pca.fit_transform(X)
    eigenvalues = pca.explained_variance_
    plt.plot(eigenvalues)
    if data_key == "fashion":
        plt.yscale("log")
    plt.xlabel("PC")
    plt.ylabel("eigenvalue")
    plt.title(f"PCA eigenvalue on {data_key} data")
    plt.savefig(f"output/pca_eigenval_{data_key}.png")
    plt.close()

    # PC
    examine_credit_cluster(
        X_pca[:, :2],
        y,
        title=f"PCA transformation on {data_key} data",
        xylabel=["PC1", "PC2"],
        fname=f"output/pca_pc_cluster_{data_key}.png",
    )

    if data_key == "fashion":
        # PC image
        plot_fashion_cluster(
            pca.components_[:25, :], range(25), fname="output/pca_pc_fashion.png"
        )

        # reconstructed image
        pca = PCA(n_components=0.95, whiten=True, random_state=0)
        X_recon = pca.inverse_transform(pca.fit_transform(X))
        print(f"...Keep {pca.n_components_} components for {data_key} data...")
        plot_fashion_cluster(X_recon, y, fname="output/pca_reconstructed_fashion.png")

        # recon vs k
        sample = []
        for nc in range(2, 201, 8):
            print(f"Reconstructing with {nc} PC...")
            pca = PCA(n_components=nc, whiten=True, random_state=0)
            X_recon = pca.inverse_transform(pca.fit_transform(X))
            sample.append(X_recon[0])
        plot_fashion_cluster(
            np.array(sample), range(25), fname="output/pca_recon_vs_k_fashion.png"
        )

    elif data_key == "credit":
        # reconstruct data
        corr_mean = []
        for nc in range(1, X.shape[1]):
            print(f"Reconstructing with {nc} PC...")
            pca = PCA(n_components=nc, whiten=True, random_state=0)
            X_recon = pca.inverse_transform(pca.fit_transform(X))
            corr = np.corrcoef(X.values.flatten(), X_recon.flatten())[0, 1]
            corr_mean.append(corr)
        plt.plot(range(1, X.shape[1]), corr_mean)
        plt.xlabel("n_components")
        plt.ylabel("corr between raw and recomposed data")
        plt.title(f"Data reconstruction quality vs PCA n_components on {data_key} data")
        plt.savefig(f"output/pca_recon_vs_k_{data_key}.png")
        plt.close()
