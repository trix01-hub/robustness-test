import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from datetime import datetime
plt.rcParams["figure.figsize"] = (10, 5)
plt.style.use('ggplot')

def df_plot(dfs, x, ys, path, ylim=None, legend_loc='best'):
    """ Plot y vs. x curves from pandas dataframe(s)

    Args:
        dfs: list of pandas dataframes
        x: str column label for x variable
        y: list of str column labels for y variable(s)
        ylim: tuple to override automatic y-axis limits
        legend_loc: str to override automatic legend placement:
            'upper left', 'lower left', 'lower right' , 'right' ,
            'center left', 'center right', 'lower center',
            'upper center', and 'center'
    """
    if ylim is not None:
        plt.ylim(ylim)
    for df, name in dfs:
        name = name.split('_')[1]
        for y in ys:
            plt.plot(df[x], df[y], linewidth=2,
                     label=name + ' ' + y.replace('_', ''))
    plt.xlabel(x.replace('_', ''))
    plt.legend(loc=legend_loc)
    plt.savefig(path + '/' + x + "--" + ys[0] +'.png')
    plt.show()


def main(env_name, folder_names, x_name):
    filepaths = [os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'log-files', env_name, 'csvs', folder_names, 'log.csv'))]
    dataframes = []
    names = []
    now = datetime.now().strftime("%Y-%m-%d_%H" + 'h' + "_%M" + 'm')
    path = os.path.join('log-files', env_name, 'charts', now, x_name)
    os.makedirs(path)
    for filepath in filepaths:
        names.append(filepath.split('\\')[7])
        dataframes.append(pd.read_csv(filepath))
    data = list(zip(dataframes, names))

    df_plot(data, x_name, ['_MeanReward'], path)
    df_plot(data, x_name, ['KL'], path)
    df_plot(data, x_name, ['ExplainedVarOld'], path)
    df_plot(data, x_name, ['PolicyLoss'], path, ylim=(-0.05, 0))

    df_plot(data, x_name, ['_mean_discrew'], path)
    df_plot(data, x_name, ['_std_discrew'], path)
    df_plot(data, x_name, ['_mean_adv'], path)
    df_plot(data, x_name, ['_std_adv'], path)

    df_plot(data, x_name, ['_mean_obs'], path)
    df_plot(data, x_name, ['_min_obs'], path)
    df_plot(data, x_name, ['_max_obs'], path)

    df_plot(data, x_name, ['_mean_act'], path)
    df_plot(data, x_name, ['_min_act'], path)
    df_plot(data, x_name, ['_max_act'], path)
    df_plot(data, x_name, ['_std_act'], path)

    df_plot(data, x_name, ['PolicyEntropy'], path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-fn', '--folder_names', type=str, help='Name of folder in log-files which contains log.csv')
    parser.add_argument('-x', '--x_name', type=str, help='Name of header in log.csv for charts',
                        default='_Episode')

    args = parser.parse_args()
    main(**vars(args))
