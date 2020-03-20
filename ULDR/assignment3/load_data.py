import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, "fashion-mnist")

from utils import mnist_reader
from utils.helper import get_sprite_image


DATA = {}
# load fashion data
X, y = mnist_reader.load_mnist("fashion-mnist/data/fashion", kind="t10k")
DATA["fashion"] = [X, y]

# load wine data
df = pd.read_csv("data/winequality-white.csv", sep=";")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
DATA["wine"] = [X, y]

# plot fashion data
X, y = DATA["fashion"]
sample = []
for i in range(10):
    X_i = X[y == i, :]
    sample.append(X_i[:10, :])
plt.imsave("output/fashion_data_sample.png", get_sprite_image(sample), cmap="gray")
plt.close()

# plot wine data
X, y = DATA["wine"]
pd.Series(y).value_counts().sort_index().plot(
    kind="bar", title="Class distribution of wine dataset"
)
plt.xlabel("class")
plt.ylabel("count")
plt.savefig("output/wine_y.png")
plt.close()
