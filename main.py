from utils import *
from visualize import *
import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env

if __name__ == '__main__':
    # TODO Guy: to add "set_display" to particle environment
    taxi_env.set_display(False)
    env_name = TAXI
    agent_name = A2C
    iteration_num = 2

    ray.init(num_gpus=NUM_GPUS, local_mode=WITH_DEBUG)
    taxi_env.set_action_abstraction(True)
    episode_reward_mean = train(env_name, agent_name, iteration_num, with_transform=False)
    # episode_reward_mean = train(env_name, agent_name, iteration_num, with_transform=True,
    #                             transform_idxes=[TAXIS_LOC_IDX, FUELS_IDX, PASS_START_LOC_IDX, PASS_DEST_IDX])
    taxi_env.set_display(True)
    evaluate()
    # evaluate([TAXIS_LOC_IDX, FUELS_IDX, PASS_START_LOC_IDX, PASS_DEST_IDX])
    ray.shutdown()

    results = [episode_reward_mean]
    names = [WITHOUT_TRANSFORM]

    plot_result_graph(agent_name, results, names, "episode_reward_mean")

    # episode_reward_mean0 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=TAXIS_LOC_IDX)
    # episode_reward_mean1 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=FUELS_IDX)
    # episode_reward_mean2 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_START_LOC_IDX)
    # episode_reward_mean3 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_DEST_IDX)
    # episode_reward_mean4 = train(env_name, agent_name, iteration_num, with_transform=True,
    #                              transform_idx=PASS_STATUS_IDX)
