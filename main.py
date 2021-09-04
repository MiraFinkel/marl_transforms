from Transforms.transform_constants import *
from experiments import *
from Transforms.single_taxi_transforms import *


def run_single_taxi_env():
    env_name = SINGLE_TAXI_EXAMPLE
    agent_name = KERAS_DQN
    num_states_in_partial_policy = 5
    num_of_epochs = 1
    num_of_episodes_per_epoch = 10000
    default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


def run_taxi_env():
    env_name = TAXI_EXAMPLE
    agent_name = KERAS_DQN
    num_states_in_partial_policy = 5
    num_of_epochs = 1
    num_of_episodes_per_epoch = 100
    different_anticipated_policy_size_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch)


def run_lunar_lander_env():
    env_name = LUNAR_LANDER
    agent_name = KERAS_DQN
    num_states_in_partial_policy = 5
    num_of_epochs = 5
    num_of_episodes_per_epoch = 500
    default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


def run_search_transform_taxi_env_example_colab():
    env_name = SEARCH_TRANSFORM_TAXI_ENV
    anticipated_policy = ANTICIPATED_POLICY
    num_of_episodes_per_epoch = ITER_NUM
    agent_name = KERAS_DQN

    original_env = get_env(SINGLE_TAXI_EXAMPLE)
    search_taxi_env = get_env(env_name)
    agent = load_existing_agent(search_taxi_env, agent_name, env_name)
    result = {}

    # evaluate the performance of the agent
    transformed_evaluation_result = rl_agent.run(agent, num_of_episodes_per_epoch, method=EVALUATE)

    # check if the anticipated policy is achieved in trans_env
    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent, anticipated_policy,
                                                                               search_taxi_env)

    result[env_name] = load_pkl_file(TRAINED_AGENT_RESULT_FILE_PATH)
    if anticipated_policy_achieved:
        result[env_name][GOT_AN_EXPLANATION] = True

    if result[env_name][GOT_AN_EXPLANATION]:
        print(f"\nexplanation found! on the {env_name} environment.")
        explanation = map_actions_to_explanation(original_env, agent, search_taxi_env, anticipated_policy)
        for transform_name, actions in explanation.items():
            print(f"transform name: {transform_name} for mapping action {actions[0]} to {actions[1]}")
        print(explanation)
    else:
        print("no explanation found:-(")


# if __name__ == '__main__':
#     agent_name = KERAS_DQN
#     env_name = APPLE_PICKING
#     new_env = get_env(APPLE_PICKING)
#
#     agent, restored = make_or_restore_model(new_env, agent_name, env_name)
#
#     print(f"\nTraining and evaluating the {agent_name} on \"{env_name}\" environment")
#     if not restored:
#         train_result = rl_agent.run(agent, ITER_NUM, method=TRAIN)
#     else:
#         train_result = None
#     print("DONE!")
