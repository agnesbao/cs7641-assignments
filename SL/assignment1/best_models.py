from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


BEST_MODELS = {
    "knn": {
        "credit": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "wine": KNeighborsClassifier(n_neighbors=33, weights="distance"),
    },
    "svm": {"credit": SVC(kernel="rbf"), "wine": SVC(kernel="rbf")},
    "mlp": {
        "credit": MLPClassifier(
            hidden_layer_sizes=(144,),
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            learning_rate_init=0.01,
        ),
        "wine": MLPClassifier(
            hidden_layer_sizes=(33,),
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            learning_rate_init=0.01,
        ),
    },
    "dt": {
        "credit": DecisionTreeClassifier(ccp_alpha=0.006),
        "wine": DecisionTreeClassifier(ccp_alpha=0.0002),
    },
    "boosting": {
        "credit": AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(ccp_alpha=0.006), n_estimators=90
        ),
        "wine": AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(ccp_alpha=0.0002), n_estimators=500
        ),
    },
}
