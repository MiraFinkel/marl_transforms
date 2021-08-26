from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from utils import *
from save_load_utils import *

ACT = "action"
IDX = "idx"
VAL = "val"


class SingleTaxiTransformSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(RELATIVE_PRECONDITIONS_PATH)
        all_single_transformed_envs = load_pkl_file(RELATIVE_SINGLE_SMALL_TAXI_TRANSFORMED_ENVS)
        for act, act_info in preconditions.not_allowed_features.items():
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    if transform_name not in all_single_transformed_envs.keys():
                        continue
                    transformed_env = all_single_transformed_envs[transform_name]
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
        basic_actions = [_ for _ in range(self.num_actions)]
        preconditions = load_pkl_file(RELATIVE_PRECONDITIONS_PATH)
        all_single_transformed_envs = load_pkl_file(RELATIVE_SINGLE_SMALL_TAXI_TRANSFORMED_ENVS)
        decoded_anticipated_states, anticipated_actions = anticipated_policy.keys(), anticipated_policy.values()
        # flatten_anticipated_actions = [item for sublist in anticipated_actions for item in sublist]
        for state, state_info in self.P.items():
            new_act = self.num_actions - 1
            align_with_anticipated, anticipated_act, _ = is_state_align_with_anticipated_policy(self, state,
                                                                                                anticipated_policy)
            for not_allowed_act, not_allowed_act_info in preconditions.not_allowed_features.items():
                for pre_idx, pre_val in not_allowed_act_info.items():
                    for val in pre_val:
                        transform_name = f"{not_allowed_act}_{pre_idx}_{val}"
                        if transform_name not in all_single_transformed_envs.keys():
                            continue
                        new_act += 1
                        transformed_env = all_single_transformed_envs[transform_name]
                        self.action_translation_dict[new_act] = {ACT: not_allowed_act, IDX: pre_idx, VAL: val}
                        if align_with_anticipated:
                            temp_info = list(transformed_env.P[state][not_allowed_act][0])
                            if new_act == 24 and state == 8950:  # TODO - its a patch! to delete!
                                temp_info[3] = True
                                self.P[state][new_act] = [tuple(temp_info)]
                            else:
                                self.P[state][new_act] = transformed_env.P[state][not_allowed_act]
                        else:
                            temp_info = list(transformed_env.P[state][not_allowed_act][0])
                            temp_info[1] = state
                            temp_info[2] = temp_info[2] if temp_info[2] < 0 else -1.0
                            self.P[state][new_act] = [tuple(temp_info)]

            if align_with_anticipated:
                next_anticipated_state = self.P[state][anticipated_act][0][1]
                if state != next_anticipated_state:
                    relevant_actions = basic_actions
                else:
                    relevant_actions = [k for (k, v) in self.action_translation_dict.items() if
                                        v[ACT] == anticipated_act]
                    relevant_actions.append(anticipated_act)
                for other_act, other_act_info in self.P[state].items():
                    if other_act not in relevant_actions:
                        temp_info = list(other_act_info[0])
                        temp_info[1] = state
                        temp_info[2] = temp_info[2] if temp_info[2] < 0 else -1.0
                        self.P[state][other_act] = [tuple(temp_info)]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


# if __name__ == '__main__':
#     anticipated_policy = ANTICIPATED_POLICY
#     search_env = SingleTaxiTransformAnticipatedSearchEnv(anticipated_policy)
    # file_name = "taxi_example_data/taxi_transformed_env/search_env_without_5.pkl"
    # save_pkl_file(file_name, search_env)
    #
    # transform_name = "search_anticipated_env"
    # save_pkl_file(transform_name, search_env)
    # file = open(file_name, "rb")
    # new_env = pickle.load(file)
    # generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
    # actions = [1, 1, 19, 7, 9, 9, 7, 24]
    # all_reward = 0
    # for act in actions:
    #     search_env.render()
    #     next_s, r, d, prob = search_env.step(act)
    #     all_reward += r
    #     print(f"state:{search_env.decode(next_s)}")
    #     print(f"reward:{r} done:{d} prob:{prob}")
    #     print(f"all_reward:{all_reward}")
    # a = 7
