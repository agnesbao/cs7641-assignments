import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, "fashion-mnist")

from utils import mnist_reader
from utils.helper import get_sprite_image


DATA = {}
# load fashion data
X, y = mnist_reader.load_mnist("fashion-mnist/data/fashion", kind="t10k")
DATA["fashion"] = [X, y]

# load credit data
df = pd.read_csv("data/dataset_31_credit-g.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X = pd.get_dummies(
    X, columns=X.select_dtypes(exclude="number").columns, drop_first=True
)
X = pd.DataFrame(data=MinMaxScaler().fit_transform(X), columns=X.columns)
y = y == "bad"
DATA["credit"] = [X, y]

# plot fashion data
X, y = DATA["fashion"]
sample = []
for i in range(10):
    X_i = X[y == i, :]
    sample.append(X_i[:10, :])
plt.imsave("output/fashion_data_sample.png", get_sprite_image(sample), cmap="gray")
plt.close()
