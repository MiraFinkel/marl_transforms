from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import numpy as np
from itertools import product
import random

NEW_MAP = [
    "+-----+",
    "|X: :X|",
    "| : : |",
    "|X:F:X|",
    "+-----+",
]


class TaxiSimpleEnv(TaxiEnv):
    def __init__(self, max_fuel=None, domain_map=None):
        super().__init__(num_taxis=1, num_passengers=1, max_fuel=None, domain_map=NEW_MAP,
                         collision_sensitive_domain=False)

    def encode(self, taxi_row, taxi_col, fuel, pass_loc_x, pass_loc_y, dest_idx_x, dest_idx_y, pass_status):
        # (self.num_rows), self.num_columns, max_fuel[0] + 1, self.num_rows, self.num_columns, self.passengers_locations, 4
        dest_idx = self._get_pass_dest_idx(dest_idx_x, dest_idx_y)

        i = taxi_row

        i *= self.num_columns
        i += taxi_col

        i *= self.max_fuel[0] + 1
        i += fuel

        i *= self.num_rows
        i += pass_loc_x

        i *= self.num_columns
        i += pass_loc_y

        i *= len(self.passengers_locations)
        i += dest_idx

        i *= 3
        i += pass_status
        return i

    def decode(self, i):
        # 4, self.passengers_locations, self.num_columns, self.num_rows, max_fuel[0] + 1, self.num_columns, self.num_rows
        j = i
        out = []

        passenger_status = [(i % 3) + 1]
        out.append(passenger_status)
        i = i // 3

        passenger_dest_idx = [self.passengers_locations[i % len(self.passengers_locations)]]
        out.append(passenger_dest_idx)
        i = i // len(self.passengers_locations)

        passenger_loc_y = i % self.num_columns
        i = i // self.num_columns
        passenger_loc_x = i % self.num_rows
        i = i // self.num_rows
        passenger_location = [[passenger_loc_x, passenger_loc_y]]
        out.append(passenger_location)

        fuel = [i % (self.max_fuel[0] + 1)]
        out.append(fuel)
        i = i // (self.max_fuel[0] + 1)

        taxi_y = i % self.num_columns
        i = i // self.num_columns
        taxi_x = i
        taxi_loc = [[taxi_x, taxi_y]]
        out.append(taxi_loc)

        assert 0 <= i < self.num_rows

        return reversed(out)

    def _get_pass_dest_idx(self, dest_idx_x, dest_idx_y):
        dest_idx = -1
        for i, loc in enumerate(self.passengers_locations):
            if dest_idx_x == loc[0] and dest_idx_y == loc[1]:
                dest_idx = i
                break
        if dest_idx == -1:
            raise Exception("no such destination!")
        return dest_idx


class TaxiSimpleExampleEnv(TaxiSimpleEnv):
    """
    This environment fixes the starting state for every episode to be:
    taxi starting location - (2,0)
    taxi fuel level - 1
    passenger starting location - (0, 0)
    passenger destination - (1, 2)
    """

    def reset(self) -> dict:
        """
        Reset the environment's state:
            - taxis coordinates - fixed.
            - refuel all taxis
            - destinations - fixed.
            - passengers locations - fixed.
            - preserve other definitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state.

        """
        # reset taxis locations
        taxis_locations = [[2, 0]]
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = 5
        self.counter = 0

        # refuel everybody
        fuels = [2 for i in range(self.num_taxis)]

        # reset passengers
        passengers_start_location = [[0, 0]]
        passengers_destinations = [[1, 2]]

        # Status of each passenger: delivered (1), in_taxi (positive number>2), waiting (2)
        passengers_status = [2 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        # resetting dones
        self.dones = {taxi_id: False for taxi_id in self.taxis_names}
        self.dones['__all__'] = False
        obs = {}
        for taxi_id in self.taxis_names:
            obs[taxi_id] = self.get_observation(self.state, taxi_id)

        return obs


class TaxiTransformedEnv(TaxiSimpleExampleEnv):

    def get_states_from_partial_obs(self, partial_obs):  # TODO Mira: add the multi agent case
        partial_obs_aligned_with_env = False
        iter_num = 100
        while not partial_obs_aligned_with_env and iter_num != 0:
            obs = self.reset()
            obs = obs[list(obs.keys())[0]].tolist()[0]  # get the observation as a list
            if self._is_aligned(obs, partial_obs):
                partial_obs_aligned_with_env = True
            iter_num -= 1

        taxi_x = [partial_obs[0]] if (partial_obs[0] is not None) else list(range(self.num_columns))
        taxi_y = [partial_obs[1]] if (partial_obs[1] is not None) else list(range(self.num_rows))
        fuel = [partial_obs[2]] if partial_obs[2] else list(range(self.max_fuel[0]))
        passenger_start_x, passenger_start_y = [obs[3]], [obs[4]]
        passenger_dest_x, passenger_dest_y = [obs[5]], [obs[6]]
        passenger_status = [partial_obs[7]] if partial_obs[7] else list(range(1, 4))
        states = list(
            product(taxi_x, taxi_y, fuel, passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y,
                    passenger_status, repeat=1))
        states = [list(state) for state in states]
        return states

    def _is_aligned(self, obs, partial_obs):
        taxi_x, taxi_y = partial_obs[0], partial_obs[1]
        passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y = self._get_passenger_info(partial_obs)
        return (taxi_x is None or passenger_start_x == obs[0]) and (
                taxi_y is None or passenger_start_x == obs[1]) and (
                passenger_start_x is None or passenger_start_x == obs[3]) and (
                passenger_start_y is None or passenger_start_y == obs[4]) and (
                passenger_dest_x is None or passenger_dest_x == obs[5]) and (
                passenger_dest_y is None or passenger_dest_y == obs[6])

    def _get_passenger_info(self, partial_obs):
        passenger_start_x, passenger_start_y = partial_obs[3], partial_obs[4]
        passenger_dest_x, passenger_dest_y = partial_obs[5], partial_obs[6]
        return passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y
