import pickle
from load_data import DATA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from scipy.stats import kurtosis


DATA_RED = {key: {} for key in DATA}

for key, val in DATA.items():
    print(f"Transforming {key} data...")
    X, y = val
    # pca
    pca = PCA(n_components=0.95, whiten=True, random_state=0)
    X_pca = pca.fit_transform(X - X.mean())
    DATA_RED[key]["pca"] = X_pca, y
    # ica
    ica = FastICA(n_components=X_pca.shape[1], whiten=True, random_state=0)
    X_ica = ica.fit_transform(X)
    DATA_RED[key]["ica"] = X_ica, y
    # rca
    rca = GaussianRandomProjection(n_components=X_pca.shape[1], random_state=0)
    X_rca = rca.fit_transform(X)
    kurt = kurtosis(X_ica)
    kurt_rank = (-kurt).argsort()

    DATA_RED[key]["rca"] = X_rca[:, kurt_rank], y
    # tsne
    tsne = TSNE(n_components=3, random_state=0, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    DATA_RED[key]["tsne"] = X_tsne, y

with open("data/dim_red_data.pkl", "wb") as f:
    pickle.dump(DATA_RED, f)
