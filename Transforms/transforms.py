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


class DimReductionMultiAgents(MappingFunction):
    def __init__(self, env, reduction_idx=DIM_REDUCTION_IDX):
        super(DimReductionMultiAgents, self).__init__(env)
        self.reduction_idx = reduction_idx
        self._mapping_dict = {}

    def mapping_step(self, state, action_dict):
        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self._get_abstract_observation(state, taxi_id)
        return obs

    def _get_abstract_observation(self, state, taxi_id):
        agent_index = self._env.taxis_names.index(taxi_id)
        taxis, fuels, passengers_start_locations, destinations, passengers_status = state
        passengers_information = self._get_passengers_information(passengers_start_locations, destinations,
                                                                  passengers_status)
        closest_taxis_indices = []
        fuels = [fuels[agent_index]]
        for i in range(self._env.num_taxis):
            if self._env.get_l1_distance(taxis[agent_index], taxis[i]) <= self._env.window_size and i != agent_index:
                closest_taxis_indices.append(i)
        if self.reduction_idx == FUELS_IDX:
            fuels = [100]
            observations = taxis[agent_index].copy()
            for i in closest_taxis_indices:
                observations += taxis[i]
        elif self.reduction_idx == TAXIS_LOC_IDX:
            observations = [0, 0]
            for i in closest_taxis_indices:
                observations += [0, 0]
        else:
            observations = taxis[agent_index].copy()
            for i in closest_taxis_indices:
                observations += taxis[i]
        observations += [0, 0] * (self._env.num_taxis - 1 - len(closest_taxis_indices)) + fuels + \
                        [0] * (self._env.num_taxis - 1) + passengers_information
        observations = np.reshape(observations, (1, len(observations)))

        return observations

    def _get_passengers_information(self, passengers_start_locations, destinations, passengers_status):
        if self.reduction_idx < 2:
            passengers_start_locations = flatten(passengers_start_locations)
            destinations = flatten(destinations)
        elif self.reduction_idx == PASS_START_LOC_IDX:
            passengers_start_locations = [0, 0] * len(passengers_start_locations)
            destinations = flatten(destinations)
        elif self.reduction_idx == PASS_DEST_IDX:
            passengers_start_locations = flatten(passengers_start_locations)
            destinations = [0, 0] * len(destinations)
        elif self.reduction_idx == PASS_STATUS_IDX:
            passengers_start_locations = flatten(passengers_start_locations)
            destinations = flatten(destinations)
            passengers_status = [0] * len(passengers_status)
        return passengers_start_locations + destinations + passengers_status

    def set_reduction_idx(self, new_idx):
        self.reduction_idx = new_idx


MAPPING_CLASS = DimReductionMultiAgents


class TransformEnvironment(MultiAgentEnv):
    def __init__(self, env_name=ENV_NAME, mapping_class=MAPPING_CLASS, **kwargs):
        self._env = TaxiEnv(kwargs)
        self._mapping_class = None
        if mapping_class:
            self._mapping_class = mapping_class(self._env)
        self.action_space = self._env.action_space
        self.obs_space = self._env.obs_space

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
