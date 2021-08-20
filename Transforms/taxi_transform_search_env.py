import pickle

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from save_load_utils import *
from train_agent import generate_agent, KERAS_DQN
from utils import *

SAVE_PATH = "taxi_example_data/"
TRANSFORMS_PATH = "taxi_example_data/taxi_transformed_env/single_transform_envs/"


class SingleTaxiTransformSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(SAVE_PATH + "taxi_example_preconditions.pkl")
        for act, act_info in preconditions.not_allowed_features.items():
            if act == 5:
                continue
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    transformed_env = load_pkl_file(TRANSFORMS_PATH + transform_name)
                    self.action_translation_dict[new_act] = transform_name
                    for s, s_info in self.P.items():
                        self.P[s][new_act] = transformed_env.P[s][act]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


class SingleTaxiTransformAnticipatedSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self, anticipated_policy):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(SAVE_PATH + "taxi_example_preconditions.pkl")
        for act, act_info in preconditions.not_allowed_features.items():
            if act == 5:
                continue
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    transformed_env = load_pkl_file(TRANSFORMS_PATH + transform_name)
                    self.action_translation_dict[new_act] = transform_name
                    for s, s_info in self.P.items():
                        if s in anticipated_policy.keys():
                            transform_info = transformed_env.P[s][act][0]
                            transform_info[0] = 0
                            for a in anticipated_policy[s]:
                                self.P[s][a] =0
                        else:
                            self.P[s][new_act] = transformed_env.P[s][act]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


if __name__ == '__main__':
    search_env = SingleTaxiTransformSearchEnv()
    file_name = "taxi_example_data/taxi_transformed_env/search_env_without_5.pkl"
    save_pkl_file(file_name, search_env)

    transform_name = "search_env_without_5"
    file = open(file_name, "rb")
    new_env = pickle.load(file)
    generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
    a = 7
