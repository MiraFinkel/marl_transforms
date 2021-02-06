import matplotlib.pyplot as plt

from constants import *


def prepare_result_graph(result, sub_plot_name=""):
    plt.plot(range(len(result)), result, label=sub_plot_name)
    plt.xlabel('epochs')
    plt.ylabel('episode_reward_mean')
    plt.ylim(-300, 500)


def plot_result_graph(agent_name, results, names, title):
    for res, name in zip(results, names):
        prepare_result_graph(res, name)
    plt.title("agent: " + agent_name + ", " + title)
    plt.legend()
    plt.show()
