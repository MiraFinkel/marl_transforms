import matplotlib.pyplot as plt

from constants import *


def prepare_result_graph(result, sub_plot_name="", x_label="", y_label=""):
    plt.plot(range(len(result)), result, label=sub_plot_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


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
    plot_success_rate_charts(names, success_rate, 'success_rate by transforms', 'success_rate')


def plot_graph_by_transform_name_and_env(agent_name, result, title=""):
    for transform_name, reward in result.items():
        prepare_result_graph(reward, sub_plot_name=transform_name, x_label="environments", y_label="mean_reward")
    show_fig(agent_name, title)


def plot_result_graph(agent_name, results, names, title):
    for res, name in zip(results, names):
        prepare_result_graph(res, sub_plot_name=name, x_label='episodes', y_label='episode_reward_mean')
    show_fig(agent_name, title)


def show_fig(agent_name, title):
    plot_name = agent_name + "_" + title
    plt.title(plot_name)
    plt.legend()
    plt.show()
    plt.savefig(plot_name + '.png')


def plot_success_rate_charts(names, success_rate, plot_name="", y_label=""):
    fig, ax = plt.subplots()
    ax.bar(names, success_rate, width=0.35)
    ax.set_ylabel(y_label)
    ax.set_title(plot_name)
    fig.align_xlabels()
    plt.show()
    plt.savefig(plot_name + '.png')
