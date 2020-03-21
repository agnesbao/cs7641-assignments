from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import plot_fashion_cluster

for data_key in DATA:
    print(f"Running PCA on {data_key} data")
    X, y = DATA[data_key]
    if data_key == "wine":
        X = StandardScaler().fit_transform(X)
    elif data_key == "fashion":
        X = X - X.mean()

    pca = PCA(whiten=True, random_state=0)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    plt.plot(eigenvalues)
    if data_key == "fashion":
        plt.yscale("log")
    plt.xlabel("PC")
    plt.ylabel("eigenvalue")
    plt.title(f"PCA eigenvalue on {data_key} data")
    plt.savefig(f"output/pca_eigenval_{data_key}.png")
    plt.close()

    if data_key == "fashion":
        # PC image
        plot_fashion_cluster(
            pca.components_[:25, :], range(25), fname="output/pca_fashion.png"
        )
        # reconstructed image
        pca = PCA(n_components=0.95, whiten=True, random_state=0)
        X_recon = pca.inverse_transform(pca.fit_transform(X))
        print(f"...Keep {pca.n_components_} components for {data_key} data...")
        plot_fashion_cluster(X_recon, y, fname="output/pca_reconstructed_fashion.png")
