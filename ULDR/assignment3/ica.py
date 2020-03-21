from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

from load_data import DATA
from examine_cluster import plot_fashion_cluster

for data_key in DATA:
    print(f"Running ICA on {data_key} data")
    X, y = DATA[data_key]
    X = StandardScaler().fit_transform(X)
    ica = FastICA(max_iter=1000, whiten=True, random_state=0)
    X_ica = ica.fit_transform(X)
    kurt = kurtosis(X_ica)
    plt.plot(kurt)
    plt.xlabel("IC")
    plt.ylabel("kurtosis")
    plt.title(f"ICA kurtosis on {data_key} data")
    plt.savefig(f"output/ica_kurtosis_{data_key}.png")
    plt.close()

    if data_key == "fashion":
        top_kurt_ind = (-kurt).argsort()
        plot_fashion_cluster(
            ica.components_[top_kurt_ind[:25], :],
            range(25),
            fname="output/ica_fashion.png",
        )
