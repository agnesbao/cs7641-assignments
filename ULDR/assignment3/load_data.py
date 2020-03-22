import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(1, "fashion-mnist")

from utils import mnist_reader
from utils.helper import get_sprite_image


DATA = {}
# load fashion data
X, y = mnist_reader.load_mnist("fashion-mnist/data/fashion", kind="t10k")
DATA["fashion"] = [X, y]
# plot fashion data
sample = []
for i in range(10):
    X_i = X[y == i, :]
    sample.append(X_i[:10, :])
plt.imsave("output/fashion_data_sample.png", get_sprite_image(sample), cmap="gray")
plt.close()


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

# use decision tree to find top features in credit data
clf = DecisionTreeClassifier(random_state=0)
clf.fit(DATA["credit"][0], DATA["credit"][1])
fi = pd.DataFrame(data=clf.feature_importances_, index=DATA["credit"][0].columns)
fi.plot(
    kind="bar",
    legend=False,
    title="Feature importance from decision tree on credit data",
)
plt.tight_layout()
plt.savefig("output/feature_importance_credit.png")
plt.close()
TOP_FEATURES = (-clf.feature_importances_).argsort()

X_weighted = X.mul(np.repeat(fi.T.values, X.shape[0], axis=0))
DATA["credit_weighted"] = X_weighted, y
