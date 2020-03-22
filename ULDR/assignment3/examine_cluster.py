import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "fashion-mnist")

from utils.helper import get_sprite_image


def examine_credit_cluster(X, labels, title, fname, xylabel=None):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=labels, s=1, alpha=0.5, cmap="coolwarm")
    plt.title(title)
    if xylabel:
        plt.xlabel(xylabel[0])
        plt.ylabel(xylabel[1])
    plt.savefig(fname)
    plt.close()


def plot_fashion_cluster(X, labels, fname):
    sample = []
    for i in np.unique(labels):
        X_i = X[labels == i, :]
        sample.append(X_i[:10, :])
    plt.imsave(fname, get_sprite_image(sample), cmap="gray")
    plt.close()
