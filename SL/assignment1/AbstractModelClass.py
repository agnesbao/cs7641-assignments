import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_plot(mean_df, ylabel, title, fname, std_df=None):
    if std_df is not None:
        mean_df.plot(yerr=std_df, title=title)
    else:
        mean_df.plot(style=".-", title=title)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("output", fname))
    plt.clf()


class _AbstractModelClass:
    def construct_model(self):
        pass

    def run_experiment(self, data, n_iter=20):
        df_list = []
        for n in range(n_iter):
            self.model.fit(data.X, data.y)
            df_list.append(pd.DataFrame(self.model.cv_results_))
        res_df = pd.concat(df_list)

        return res_df
