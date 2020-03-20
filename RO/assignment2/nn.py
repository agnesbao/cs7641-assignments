import os
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import pandas as pd
from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# load data
dat = pd.read_csv(os.path.join("data", "dataset_31_credit-g.csv"))
dat = pd.get_dummies(dat, drop_first=True)
X = dat.iloc[:, :-1]
y = 1 - dat.iloc[:, -1]
y.columns = ["bad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# scale input
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
algo_list = [
    "random_hill_climb",
    "simulated_annealing",
    "genetic_alg",
    "gradient_descent",
]

curve_list = []
acc_train_list = []
acc_test_list = []
prec_train_list = []
prec_test_list = []
recall_train_list = []
recall_test_list = []
f1_train_list = []
f1_test_list = []
time_list = []

for algo in algo_list:
    print("Running " + algo)

    nn = mlrose.NeuralNetwork(
        hidden_nodes=[100],
        activation="relu",
        algorithm=algo,
        clip_max=1,
        max_iters=200,
        learning_rate=0.001,
        random_state=42,
        curve=True,
    )

    t1 = process_time()
    nn.fit(X_train, y_train)
    t2 = process_time()
    curve = nn.fitness_curve
    time_list.append((t2 - t1) / len(curve))

    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_train = precision_score(y_train, y_pred_train)
    prec_test = precision_score(y_test, y_pred_test)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    if algo == "gradient_descent":
        curve *= -1
    curve_list.append(curve)
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)
    prec_train_list.append(prec_train)
    prec_test_list.append(prec_test)
    recall_train_list.append(recall_train)
    recall_test_list.append(recall_test)
    f1_train_list.append(f1_train)
    f1_test_list.append(f1_test)

acc_df = pd.DataFrame(
    data={"accuracy_train": acc_train_list, "accuracy_test": acc_test_list},
    index=algo_list,
)
print(acc_df)
acc_df.plot(style=".-")
plt.ylabel("accuracy")
plt.title("Neural Network Weight Optimization: accuracy vs algorithms")
plt.savefig("output/nn_acc.png")
plt.close()

f1_df = pd.DataFrame(
    data={"f1_train": f1_train_list, "f1_test": f1_test_list}, index=algo_list,
)
print(f1_df)
f1_df.plot(style=".-")
plt.ylabel("f1")
plt.title("Neural Network Weight Optimization: f1-score vs algorithms")
plt.savefig("output/nn_f1.png")
plt.close()

prec_df = pd.DataFrame(
    data={"prec_train": prec_train_list, "prec_test": prec_test_list}, index=algo_list,
)
print(prec_df)
prec_df.plot(style=".-")
plt.ylabel("precision")
plt.title("Neural Network Weight Optimization: precision vs algorithms")
plt.savefig("output/nn_prec.png")
plt.close()

recall_df = pd.DataFrame(
    data={"recall_train": recall_train_list, "recall_test": recall_test_list},
    index=algo_list,
)
print(recall_df)
recall_df.plot(style=".-")
plt.ylabel("precision")
plt.title("Neural Network Weight Optimization: recall vs algorithms")
plt.savefig("output/nn_recall.png")
plt.close()

df = pd.DataFrame(curve_list).transpose()
df.columns = algo_list
df.plot(
    logy=True,
    subplots=True,
    layout=(2, 2),
    figsize=(10, 4),
    title="Neural Network: Loss (fitness) curve vs algorithms",
)
plt.xlabel("Iteration")
plt.ylabel("Log-loss")
plt.savefig("output/nn_algo.png")
plt.close()

print(time_list)
