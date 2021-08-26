from experiments import *
from save_load_utils import *
from constants import *
from save_load_utils import load_transform_by_name


def generate_agent(env_name, agent_name, num_of_episodes, transformed_env, transform_name):
    original_env = get_env(env_name)
    result = dict()
    anticipated_policy = ANTICIPATED_POLICY

    explanation = []

    result, explanation = create_run_and_evaluate_agent(original_env, transformed_env, agent_name, transform_name,
                                                        num_of_episodes, anticipated_policy, result, explanation)

    if explanation is None or len(explanation) == 0:
        print(f"{transform_name} is not your answer! it is not an explanation")
    else:
        print(f"explanation found {explanation}!!")

    dir_name = TRAINED_AGENTS_DIR_PATH + transform_name + "_" + agent_name + "_200000"
    make_dir(dir_name)
    save_pkl_file(dir_name + "/" + transform_name + "_" + agent_name + "_result", result)
    save_pkl_file(dir_name + "/" + transform_name + "_" + agent_name + "_explanation", explanation)


if __name__ == '__main__':
    # path = sys.argv[1:][0]
    path = "0_(0,)_[4].pkl"
    file_name = os.path.basename(path)
    transform_dir = "single_transform_envs/"
    transform_name, new_env = load_transform_by_name(transform_dir + file_name)
    generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
