import pickle

from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from save_load_utils import *
from Testing.train_agent import generate_agent, KERAS_DQN
from utils import *

SAVE_PATH = "taxi_example_data/"
TRANSFORMS_PATH = "taxi_example_data/taxi_transformed_env/single_transform_envs/"


class SingleTaxiTransformSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(SAVE_PATH + "taxi_example_preconditions.pkl")
        all_single_transformed_envs = load_pkl_file(SAVE_PATH + "all_single_transformed_envs.pkl")
        for act, act_info in preconditions.not_allowed_features.items():
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    transformed_env = all_single_transformed_envs[transform_name]
                    self.action_translation_dict[new_act] = transform_name
                    for s, s_info in self.P.items():
                        self.P[s][new_act] = transformed_env.P[s][act]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


class SingleTaxiTransformAnticipatedSearchEnv(SingleTaxiTransformSearchEnv):
    def __init__(self, anticipated_policy):
        super().__init__()
        for s, s_info in self.P.items():
            align_with_anticipated_policy, anticipated_action = self.is_state_align_with_anticipated_policy(s, anticipated_policy)
            if align_with_anticipated_policy:
                for cur_act, cur_info in self.P[s].items():
                    if cur_act == anticipated_action:
                        self.P[s][cur_act] = transformed_env.P[s][anticipated_action][0]
                    else:
                        info_to_replace = list(cur_info[0])
                        info_to_replace[0] = 0.0
                        self.P[s][cur_act] = [tuple(info_to_replace)]
            else:
                self.P[s][new_act] = transformed_env.P[s][act]

    def is_state_align_with_anticipated_policy(self, state, anticipated_policy):
        for anticipated_state, anticipated_action in anticipated_policy.items():
            if self.is_state_align_with_anticipated_state(state, anticipated_state):
                return True, anticipated_action
        return False, None

    def is_state_align_with_anticipated_state(self, state, anticipated_state):
        anticipated_state = list(anticipated_state)
        decoded_state = self.decode(state)
        for (anticipated_feature, feature) in zip(anticipated_state, decoded_state):
            if anticipated_feature and feature != anticipated_feature:
                return False
        return True


# if __name__ == '__main__':
#     anticipated_policy = {(2, 0, 0, 3, None): [1],
#                           (1, 0, 0, 3, None): [1],
#                           (0, 0, 0, 3, None): [4]}
#     search_env = SingleTaxiTransformAnticipatedSearchEnv(anticipated_policy)
    # file_name = "taxi_example_data/taxi_transformed_env/search_env_without_5.pkl"
    # save_pkl_file(file_name, search_env)
    #
    # transform_name = "search_env_without_5"
    # file = open(file_name, "rb")
    # new_env = pickle.load(file)
    # generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
    a = 7
