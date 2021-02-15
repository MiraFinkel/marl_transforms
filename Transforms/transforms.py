from copy import deepcopy
from abc import ABC, abstractmethod
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
from ray.rllib.env import MultiAgentEnv
import numpy as np
from Transforms.transform_constans import *


def flatten(x):
    return [item for sub in list(x) for item in list(sub)]


class MappingFunction(ABC):
    @abstractmethod
    def __init__(self, env):
        """
        Initialize the data structure of the mapping
        """
        self._env = deepcopy(env)

    @abstractmethod
    def mapping_step(self, state, action):
        """
        Step in the mapped environment
        """
        pass

    def _get_transition_info(self, state, action):
        transitions = self._env.P[state][action]
        if len(transitions) == 1:
            i = 0
        else:
            i = self._env.categorical_sample([t[0] for t in transitions], self._env.np_random)
        p, s, r, d = transitions[i]
        return p, s, r, d


global_reduction_idxes = None


class DimReductionMultiAgents(MappingFunction):
    """
    A wrapper class to the Multi-Taxi environment class.
    This abstraction exposed to the agent the whole state except for the features in place of the reduction indexes.
    """

    def __init__(self, env, reduction_idxes=None):
        """
        The constructor of the dimension reduction class.
        :param env: The original environment
        :param reduction_idxes: list - A list of indexes of the state that the agent won't see.
        """
        super(DimReductionMultiAgents, self).__init__(env)
        if reduction_idxes is None:
            reduction_idxes = global_reduction_idxes
        self.reduction_idxes = reduction_idxes
        self._mapping_dict = {}

    def mapping_step(self, state, action_dict):
        """
        A wrapper function for an abstract step in the transformed environment.
        Executing a list of actions (action for each taxi) at the domain current state.
        :param state: current state
        :param action_dict: {taxi_name: action} - action of specific taxis to take on the step
        :return: The observation after the transform.
        """
        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self._get_abstract_observation(state, taxi_id)
        return obs

    def _get_abstract_observation(self, state, taxi_id):
        """

        :param state:
        :param taxi_id:
        :return:
        """
        agent_index = self._env.taxis_names.index(taxi_id)
        taxis, fuels, passengers_start_locations, destinations, passengers_status = state
        passengers_information = [flatten(passengers_start_locations), flatten(destinations), passengers_status]
        closest_taxis_indices, observations, fuels = self._get_original_info(taxis, agent_index, fuels)

        self.reduction_idxes = self.reduction_idxes if self.reduction_idxes else []
        for idx in self.reduction_idxes:
            passengers_information = self._get_passengers_information(passengers_information, idx)
            if idx == FUELS_IDX:
                fuels = [self._env.max_fuel[agent_index]]
            elif idx == TAXIS_LOC_IDX:
                observations = [0, 0]
                for _ in closest_taxis_indices:
                    observations += [0, 0]
        passengers_information = passengers_information[0] + passengers_information[1] + passengers_information[2]
        observations += [0, 0] * (self._env.num_taxis - 1 - len(closest_taxis_indices)) + fuels + \
                        [0] * (self._env.num_taxis - 1) + passengers_information
        observations = np.reshape(observations, (1, len(observations)))
        return observations

    def _get_passengers_information(self, passengers_information, idx):
        """
        Recive a index that need to be ignored and return the reduced observation
        :param passengers_information: List with the original passengers information
        :return: Reduced passengers information
        """
        if idx == PASS_START_LOC_IDX:
            passengers_information[0] = [0] * len(passengers_information[0])
        elif idx == PASS_DEST_IDX:
            passengers_information[1] = [0] * len(passengers_information[1])
        elif idx == PASS_STATUS_IDX:
            passengers_information[2] = [0] * len(passengers_information[2])
        return passengers_information

    def _get_original_info(self, taxis, agent_index, fuels):
        closest_taxis_indices = []
        for i in range(self._env.num_taxis):
            if self._env.get_l1_distance(taxis[agent_index], taxis[i]) <= self._env.window_size and i != agent_index:
                closest_taxis_indices.append(i)
        observations = taxis[agent_index].copy()
        for i in closest_taxis_indices:
            observations += taxis[i]
        fuels = [fuels[agent_index]]
        return closest_taxis_indices, observations, fuels

    def set_reduction_idx(self, new_idxes):
        self.reduction_idxes = new_idxes
        global global_reduction_idxes
        global_reduction_idxes = new_idxes


MAPPING_CLASS = DimReductionMultiAgents


class TransformEnvironment(MultiAgentEnv):
    def __init__(self, mapping_class=MAPPING_CLASS, **kwargs):
        self._mapping_class = None
        if mapping_class:
            self._mapping_class = mapping_class
        self._env = TaxiEnv(kwargs)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        """
          Resets the current state to the start state
        """
        return self._env.reset()

    def render(self):
        """
        Renders the environment.
        """
        self._env.render()

    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns (new_obs, reward, done, info)
        """
        cur_s = self._env.state
        new_obs, reward, done, info = self._env.step(action)
        if self._mapping_class:
            abstract_obs = self._mapping_class.mapping_step(cur_s, action)
            new_obs = abstract_obs
        return new_obs, reward, done, info



