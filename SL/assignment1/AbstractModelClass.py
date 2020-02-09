import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def generate_plot(mean_df, ylabel, fname, std_df=None, **kwargs):
    if std_df is not None:
        mean_df.plot(yerr=std_df, **kwargs)
    else:
        mean_df.plot(style=".-", **kwargs)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("output", fname))
    plt.clf()
    plt.close('all')


class _AbstractModelClass:
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=3, shuffle=True)
        self.scoring = "accuracy"

    def construct_model(self, **kwargs):
        pass

    def run_experiment(self, data, n_iter=20):
        random.seed(0)
        df_list = []
        for n in range(n_iter):
            print(f"iter = {str(n)}")
            self.model.fit(data.X_train, data.y_train)
            df_list.append(pd.DataFrame(self.model.cv_results_))
        res_df = pd.concat(df_list)
        res_df.to_csv(os.path.join("data", f"{data.data_name}_{self.algo_name}.csv"))
        return res_df
