import matplotlib.pyplot as plt

from constants import *


def prepare_result_graph(result, sub_plot_name=""):
    plt.plot(range(len(result)), result, label=sub_plot_name)
    plt.xlabel('episodes')
    plt.ylabel('episode_reward_mean')


def plot_result_graphs(agent_name, result):
    names = []
    train_episode_reward_mean = []
    evaluate_episode_reward_mean = []
    success_rate = []
    for name, res in result.items():
        names.append(name)
        train_episode_reward_mean.append(res["train_episode_reward_mean"])
        evaluate_episode_reward_mean.append(res["evaluate_episode_reward_mean"])
        success_rate.append(res["success_rate"])

    plot_result_graph(agent_name, train_episode_reward_mean, names, "train_episode_reward_mean")
    plot_result_graph(agent_name, evaluate_episode_reward_mean, names, "evaluate_episode_reward_mean")
    plot_success_rate_charts(names, success_rate)


def plot_result_graph(agent_name, results, names, title):
    for res, name in zip(results, names):
        prepare_result_graph(res, name)
    plt.title("agent: " + agent_name + ", " + title)
    plt.legend()
    plt.show()


def plot_success_rate_charts(names, success_rate):
    fig, ax = plt.subplots()
    ax.bar(names, success_rate, width=0.35)
    ax.set_ylabel('success_rate')
    ax.set_title('success_rate by transforms')
    ax.legend()
    fig.align_xlabels()
    plt.show()

