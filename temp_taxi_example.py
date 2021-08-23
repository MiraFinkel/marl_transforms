from experiments import *
from Transforms.single_taxi_transforms import *

AGENT_DATA_PATH = "Agents/TrainedAgents/"
TRAINED_AGENTS_DIR_PATH = AGENT_DATA_PATH + "trained_models/"


def main():
    # define the environment
    env_name = SINGLE_TAXI_EXAMPLE
    agent_name = KERAS_DQN
    num_of_episodes = 200000
    results = {}
    original_env = get_env(env_name)

    # anticipated policy is <state, action> pairs
    anticipated_policy = {(2, 0, 0, 3, None): [1],
                          (1, 0, 0, 3, None): [1],
                          (0, 0, 0, 3, None): [4],
                          (0, 0, 4, 3, None): [0],
                          (1, 0, 4, 3, None): [2],
                          (1, 1, 4, 3, None): [2],
                          (1, 2, 4, 3, None): [0],
                          (2, 2, 4, 3, None): [5]}

    new_agent = load_existing_agent(original_env, agent_name, ORIGINAL_ENV, TRAINED_AGENTS_DIR_PATH)

    evaluation_result = rl_agent.run(new_agent, 5, method=EVALUATE, print_process=False, visualize=False)

    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, new_agent,
                                                                               anticipated_policy)
    if anticipated_policy_achieved:
        print("The algorithm achieved the policy. We finished our work.")

    transforms = load_existing_transforms_from_dir()
    # t_name, t_env = load_transform_by_name("0_(4,)_[0]_1_(4,)_[0]_2_(4,)_[0]_4_(4,)_[0]_5_(4,)_[0].pkl", dir_name="Transforms/taxi_example_data/taxi_transformed_env/")
    # transforms = {0: (t_name, t_env)}
    explanations = []

    for params, (transform_name, transformed_env) in transforms.items():
        # create agent
        print(f"\nEvaluating agent on the transformed environment: {transform_name}")
        agent = load_existing_agent(transformed_env, agent_name, transform_name, TRAINED_AGENTS_DIR_PATH)
        if agent is None:
            continue
        transformed_train_result, explanation = load_existing_results(agent_name, transform_name, num_of_episodes)
        # evaluate the performance of the agent
        transformed_evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE, print_process=True,
                                                     visualize=False)

        # check if the anticipated policy is achieved in trans_env
        anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent,
                                                                                   anticipated_policy)

        results[transform_name] = {EVALUATION_RESULTS: transformed_evaluation_result,
                                   TRAINING_RESULTS: transformed_train_result,
                                   SUCCESS_RATE: success_rate,
                                   GOT_AN_EXPLANATION: False}
        if anticipated_policy_achieved:
            print(f"Got an explanation: {transform_name}")
            explanations.append(transform_name)
            results[transform_name][GOT_AN_EXPLANATION] = True

    if explanations is None or len(explanations) == 0:
        print("No explanation found! :-(")
    else:
        print(f"Explanations found: {explanations}")

    success_rates = [v[SUCCESS_RATE] for v in results.values()]
    names = [_ for _ in range(len(success_rates))]
    names_translation_dict = dict()
    for i, k in enumerate(results.keys()):
        names_translation_dict[i] = k
    fig, ax = plt.subplots()
    ax.bar(names, success_rates)
    ax.set_xticks(np.arange(len(names)))
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate of the different transformed environments")
    plt.show()
    a = 7

# main()
