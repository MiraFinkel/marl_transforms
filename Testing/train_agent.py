from experiments import *
from save_load_utils import *

DIR_PATH = "Agents/TrainedAgents/"


def generate_agent(env_name, agent_name, num_of_episodes, transformed_env, transform_name):
    original_env = get_env(env_name)
    result = dict()
    anticipated_policy = dict()

    anticipated_policy[(2, 0, 0, 3, None)] = [1]
    anticipated_policy[(1, 0, 0, 3, None)] = [1]
    anticipated_policy[(0, 0, 0, 3, None)] = [4]

    explanation = []

    result, explanation = create_run_and_evaluate_agent(original_env, transformed_env, agent_name, transform_name,
                                                        num_of_episodes, anticipated_policy, result, explanation)

    if explanation is None or len(explanation) == 0:
        print(f"{transform_name} is not your answer! it is not an explanation")
    else:
        print(f"explanation found {explanation}!!")

    dir_name = DIR_PATH + transform_name + "_" + agent_name + "_200000"
    make_dir(dir_name)
    save_pkl_file(dir_name + "/" + transform_name + "_" + agent_name + "_result", result)
    save_pkl_file(dir_name + "/" + transform_name + "_" + agent_name + "_explanation", explanation)


# if __name__ == '__main__':
#     path = sys.argv[1:][0]
#     file_name = os.path.basename(path)
#     transform_name, new_env = load_transform_by_name(file_name)
#     generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
