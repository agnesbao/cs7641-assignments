from sklearn.decomposition import PCA

for data_key in DATA:
    print(f"Running PCA on {data_key} data")
    X, y = DATA[data_key]
    if data_key == "wine":
        X = StandardScaler().fit_transform(X)
    pca = PCA(whiten=True)
    X_new = pca.fit_transform(X)
    plt.plot(pca.explained_variance_ratio_)
    plt.clf()
