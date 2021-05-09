import matplotlib.pyplot as plt
from qbstyles import mpl_style
from constants import *

mpl_style(dark=False, minor_ticks=True)
figure_size = (50, 20)
legend_alpha = 0.5


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


def plot_all_success_rate_charts(result, save_folder):
    all_success_rate = {}
    for res in result:
        for name, res_by_name in res.items():
            if name not in all_success_rate:
                all_success_rate[name] = [res_by_name[SUCCESS_RATE]]
            else:
                all_success_rate[name].append(res_by_name[SUCCESS_RATE])
    all_success_rate_mean = {}
    for name, res in all_success_rate.items():
        success_rate = np.array(res)
        success_rate_mean = success_rate.mean()
        all_success_rate_mean[name] = success_rate_mean
    names = list(all_success_rate_mean.keys())
    success_rate_means = list(all_success_rate_mean.values())
    for i in range(32):
        plot_success_rate_charts(names[i * 32: (i + 1) * 32], success_rate_means[i * 32: (i + 1) * 32],
                                 plot_name=str(i) + "_chart_graph", save_folder=save_folder)


def plot_success_rate_charts(names, success_rate, plot_name="", save_folder=None, y_label=""):
    fig, ax = plt.subplots()
    ax.bar(names, success_rate)
    ax.set_ylabel(y_label)
    ax.set_title(plot_name)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.align_labels()
    plt.show()
    plt.savefig('output/' + plot_name + '.png')


def get_plot(plt, x, stats, name, color, which_results, line="-"):
    stats_ = stats
    mean, std, q1, q3 = process_stats(stats_, name, which_results)
    mean, std, q1, q3 = mean[:len(x)], std[:len(x)], q1[:len(x)], q3[:len(x)]
    # mean_smoothed = pd.Series(mean).rolling(5, min_periods=5).mean()
    cumsum = type == 'cumsum'
    if cumsum:
        plt.plot(x, np.cumsum(mean), line, color=color)
    else:
        plt.plot(x, mean, line, color=color)
        plt.fill_between(x, q1, q3, color=color, alpha=0.2)


def process_stats(result, name, which_results, reward=True):
    '''
    returns: episode_lengths, episode_rewards, None, None or
             mean_episode_lengths, mean_episode_rewards, std_episode_lengths, std_episode_rewards
    '''

    if len(result) == 1:
        return result[0][name][which_results][EPISODE_STEP_NUM_MEAN], result[0][name][which_results][
            EPISODE_REWARD_MEAN], [0], [0]
    elif len(result) > 1:
        all_episode_lengths = []
        all_episode_variance = []
        all_episode_rewards = []
        all_episode_rewards = []

        for s in result:
            all_episode_lengths.append(s[name][which_results][EPISODE_STEP_NUM_MEAN])
            all_episode_rewards.append(s[name][which_results][EPISODE_REWARD_MEAN])

        all_episode_lengths = np.array(all_episode_lengths)
        all_episode_rewards = np.array(all_episode_rewards)

        if reward:
            q1 = np.percentile(all_episode_rewards, 25, axis=0, interpolation='midpoint')
            q3 = np.percentile(all_episode_rewards, 75, axis=0, interpolation='midpoint')
            return all_episode_rewards.mean(axis=0), all_episode_rewards.std(axis=0), q1, q3

        else:
            q1 = np.percentile(all_episode_lengths, 25, axis=0, interpolation='midpoint')
            q3 = np.percentile(all_episode_lengths, 75, axis=0, interpolation='midpoint')
            return all_episode_lengths.mean(axis=0), all_episode_lengths.std(axis=0), q1, q3


def plot_reward_graph(result, which_results, save_folder):
    max_length = len(result[0][ORIGINAL_ENV][TRAINING_RESULTS][EPISODE_REWARD_MEAN])
    epochs_num = len(result)
    fig1, ax = plt.subplots(figsize=figure_size)
    x = np.arange(1, max_length + 1)
    transforms = []
    for transform_name, res in result[0].items():
        get_plot(plt, x, result, transform_name, None, which_results)
        transforms.append(transform_name)
    plt.legend(transforms, prop={'size': 8}, ncol=10, framealpha=legend_alpha)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=max_length)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(which_results)
    plt.show()
    plt.savefig('output/' + which_results + EPISODE_REWARD_MEAN + '.png')


def plot_results(result, save_folder):
    plot_reward_graph(result, TRAINING_RESULTS, save_folder)
    plot_reward_graph(result, EVALUATION_RESULTS, save_folder)
    plot_all_success_rate_charts(result, save_folder)

    # file_name = '_' + ("prefix" + "Episode Reward over Time" + "postfix").replace(' ', '_').replace(':', '_')
    # fig1.savefig(save_folder + file_name)
