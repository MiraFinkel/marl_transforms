import matplotlib.pyplot as plt

from constants import *


def prepare_result_graph(result, sub_plot_name=""):
    # plt.plot(range(len(result['hist_stats']['episode_reward'])), result['hist_stats']['episode_reward'],
    #          label=sub_plot_name)
    plt.plot(range(len(result)), result, label=sub_plot_name)
    plt.xlabel('epochs')
    plt.ylabel('episode_reward_mean')
    plt.ylim(-300, 500)


def plot_result_graph(results, names, title):
    for res, name in zip(results, names):
        prepare_result_graph(res, name)
    plt.title("agent: " + "pg" + ", " + title)
    plt.legend()
    plt.show()
