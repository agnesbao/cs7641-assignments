import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_plot(mean_df, ylabel, title, fname, std_df=None, ylim=None):
    if std_df is not None:
        mean_df.plot(yerr=std_df, title=title, ylim=ylim)
    else:
        mean_df.plot(style=".-", title=title, ylim=ylim)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join("output", fname))
    plt.clf()


class _AbstractModelClass:
    def construct_model(self, **kwargs):
        pass

    def run_experiment(self, data, n_iter=20):
        df_list = []
        for n in range(n_iter):
            print(f"iter = {str(n)}\n")
            self.model.fit(data.X, data.y)
            df_list.append(pd.DataFrame(self.model.cv_results_))
        res_df = pd.concat(df_list)
        res_df.to_csv(os.path.join("data", f"{data.data_name}_{self.algo_name}.csv"))
        return res_df
