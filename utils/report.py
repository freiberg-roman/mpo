import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def create_graphs(
        data_path,
        plot=[('Epoch', 'AverageTestEpRet')],
        save_to=None,
        file_name='full_report'):
    # read data
    df = pd.read_table(data_path)

    # setting for plots
    fig, axes = plt.subplots(len(plot), 1, figsize=(10, 5 * len(plot)))

    sns.set(style='darkgrid', font_scale=1.5)
    for i, (x, y) in enumerate(plot):
        # plot tuple
        sns.lineplot(data=df, ax=axes[i], x=x, y=y)
        axes[i].set_title(y)

        # in case of too large values
        if np.max(np.asarray(df[x])) > 5e3:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout(pad=0.5)

    plt.savefig(save_to + "/" + file_name)
