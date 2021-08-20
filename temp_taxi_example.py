from experiments import *
from Transforms.single_taxi_transforms import *

AGENT_DATA_PATH = "Agents/TrainedAgents/"
TRAINED_AGENTS_DIR_PATH = AGENT_DATA_PATH + "trained_models/"

# define the environment
env_name = SINGLE_TAXI_EXAMPLE
agent_name = KERAS_DQN
num_of_episodes = 200000
results = {}
# get the environment

transform_name = ORIGINAL_ENV
original_env = get_env(env_name)

# anticipated policy is <state, action> pairs
anticipated_policy = {(2, 0, 0, 3, None): [1],
                      (1, 0, 0, 3, None): [1],
                      (0, 0, 0, 3, None): [4]}

new_agent = load_existing_agent(original_env, agent_name, ORIGINAL_ENV, TRAINED_AGENTS_DIR_PATH)

evaluation_result = rl_agent.run(new_agent, 5, method=EVALUATE)

anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, new_agent, anticipated_policy)
if anticipated_policy_achieved:
    print("The algorithm achieved the policy. We finished our work.")

transforms = load_existing_transforms_by_anticipated_policy(env_name, anticipated_policy)
explanations = []

for params, (transform_name, transformed_env) in transforms.items():
    # create agent
    print(f"\nEvaluating agent on the transformed environment: {transform_name}")
    agent = load_existing_agent(transformed_env, agent_name, transform_name, TRAINED_AGENTS_DIR_PATH)
    if agent is None:
        continue
    transformed_train_result, explanation = load_existing_results(agent_name, transform_name, num_of_episodes)
    # evaluate the performance of the agent
    transformed_evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE, print_process=False, visualize=False)

    # check if the anticipated policy is achieved in trans_env
    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent,
                                                                               anticipated_policy)

    results[transform_name] = {EVALUATION_RESULTS: transformed_evaluation_result,
                               TRAINING_RESULTS: transformed_train_result,
                               SUCCESS_RATE: success_rate,
                               GOT_AN_EXPLANATION: False}
    if anticipated_policy_achieved:
        print(f"Got an explanation: {explanation}")
        explanations.append(transform_name)
        results[transform_name][GOT_AN_EXPLANATION] = True

if explanations is None or len(explanations) == 0:
    print("No explanation found! :-(")
else:
    print(f"Explanations found: {explanations}")
