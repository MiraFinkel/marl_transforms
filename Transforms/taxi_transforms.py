from itertools import product
import numpy as np
from ray.rllib.env import MultiAgentEnv
import random
from Environments.MultiTaxiEnv.multitaxienv.config import TAXI_ENVIROMENT_REWARDS
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv, display

number_of_agents = 1

NEW_MAP = [
    "+-------+",
    "|X: : :X|",
    "| : : : |",
    "| : : : |",
    "|X: : :X|",
    "+-------+",
]

NEW_REWARD = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-15,
    bad_dropoff=-15,
    bad_refuel=-10,
    bad_fuel=-50,
    pickup=500,
    standby_engine_off=-1,
    turn_engine_on=-10e6,
    turn_engine_off=-10e6,
    standby_engine_on=-1,
    intermediate_dropoff=50,
    final_dropoff=1000,
    hit_wall=-2,
    collision=-35,
    collided=-20,
    unrelated_action=-15,
)


def set_number_of_agents(val):
    global number_of_agents
    number_of_agents = val


class TaxiSimpleEnv(TaxiEnv):
    def __init__(self, max_fuel=None, domain_map=None):
        super().__init__(num_taxis=1, num_passengers=1, max_fuel=None, domain_map=NEW_MAP,
                         collision_sensitive_domain=False)


class TaxiTransformedEnv(TaxiSimpleEnv):

    def get_states_from_partial_obs(self, partial_obs):  # TODO Mira: add the multi agent case
        partial_obs_aligned_with_env = False
        iter_num = 100
        while not partial_obs_aligned_with_env and iter_num != 0:
            obs = self.reset()
            obs = obs[list(obs.keys())[0]].tolist()[0]  # get the observation as a list
            if self._is_aligned(obs, partial_obs):
                partial_obs_aligned_with_env = True
            iter_num -= 1

        taxi_x = [partial_obs[0]] if partial_obs[0] else list(range(self.num_columns))
        taxi_y = [partial_obs[1]] if partial_obs[1] else list(range(self.num_rows))
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
        passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y = self._get_passenger_info(partial_obs)
        return (passenger_start_x is None or passenger_start_x == obs[3]) and (
                passenger_start_y is None or passenger_start_y == obs[4]) and (
                       passenger_dest_x is None or passenger_dest_x == obs[5]) and (
                       passenger_dest_y is None or passenger_dest_y == obs[6])

    def _get_passenger_info(self, partial_obs):
        passenger_start_x, passenger_start_y = partial_obs[3], partial_obs[4]
        passenger_dest_x, passenger_dest_y = partial_obs[5], partial_obs[6]
        return passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y


class TaxiSmallMapTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        super().__init__(domain_map=NEW_MAP)


class TaxiNoWallsTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        super().__init__(**kwargs)

    def _take_movement(self, action: str, row: int, col: int) -> (bool, int, int):
        """
        Takes a movement with regard to a specific location of a taxi,
        can take action even though there a wall in the direction of the action.
        Args:
            action: direction to move
            row: current row
            col: current col

        Returns: if moved (false if there is a wall), new row, new col

        """
        moved = False
        new_row, new_col = row, col
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        if action == 'south':  # south
            if row != max_row:
                moved = True
            new_row = min(row + 1, max_row)
        elif action == 'north':  # north
            if row != 0:
                moved = True
            new_row = max(row - 1, 0)
        if action == 'east':  # east
            if col != max_col:
                moved = True
            new_col = min(col + 1, max_col)
        elif action == 'west':  # west
            if col != 0:
                moved = True
            new_col = max(col - 1, 0)

        return moved, new_row, new_col


class TaxiInfiniteFuelTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        super().__init__(**kwargs)

    def _update_movement_wrt_fuel(self, taxi: int, taxis_locations: list, wanted_row: int, wanted_col: int,
                                  reward: int, fuel: int) -> (int, int, list):
        """
        Given that a taxi would like to move - check the fuel accordingly and update reward and location.
        Args:
            taxi: index of the taxi
            taxis_locations: list of current locations (prior to movement)
            wanted_row: row after movement
            wanted_col: col after movement
            reward: current reward
            fuel: current fuel

        Returns: updated_reward, updated fuel, updated_taxis_locations

        """
        taxis_locations[taxi] = [wanted_row, wanted_col]
        return reward, fuel, taxis_locations


class TaxiRewardTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        super().__init__(**kwargs)
        self.reward_dict = NEW_REWARD
        self.display = False

    def set_reward_dict(self, new_rewards):
        self.reward_dict = new_rewards

    def set_display(self, val):
        self.display = val

    def step(self, action_dict: dict) -> (dict, dict, dict, dict):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.
        Using manual reward.

        Args:
            action_dict: {taxi_name: action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """

        rewards = {}
        self.counter += 1
        if self.counter >= 90:
            self.window_size = 3

        # Main of the function, for each taxi-i act on action[i]
        for taxi_name, action_list in action_dict.items():
            # meta operations on the type of the action
            action_list = self._get_action_list(action_list)

            for action in action_list:
                taxi = self.taxis_names.index(taxi_name)
                reward = self.partial_closest_path_reward('step')  # Default reward
                moved = False  # Indicator variable for later use

                # taxi locations: [i, j]
                # fuels: int
                # passengers_start_locations and destinations: [[i, j] ... [i, j]]
                # passengers_status: [[1, 2, taxi_index+2] ... [1, 2, taxi_index+2]], 1 - delivered
                taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state

                if all(list(self.dones.values())):
                    continue

                # If taxi is collided, it can't perform a step
                if self.collided[taxi] == 1:
                    rewards[taxi_name] = self.partial_closest_path_reward('collided')
                    self.dones[taxi_name] = True
                    continue

                # If the taxi is out of fuel, it can't perform a step
                if fuels[taxi] == 0 and not self.at_valid_fuel_station(taxi, taxis_locations):
                    rewards[taxi_name] = self.partial_closest_path_reward('bad_fuel')
                    self.dones[taxi_name] = True
                    continue

                taxi_location = taxis_locations[taxi]
                row, col = taxi_location

                fuel = fuels[taxi]
                is_taxi_engine_on = self.engine_status_list[taxi]
                _, index_action_dictionary = self.get_available_actions_dictionary()

                if not is_taxi_engine_on:  # Engine is off
                    # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                    reward = self._engine_is_off_actions(index_action_dictionary[action], taxi)

                else:  # Engine is on
                    # Binding
                    if index_action_dictionary[action] == 'bind':
                        self.bounded = False
                        reward = self.partial_closest_path_reward('bind')

                    # Movement
                    if index_action_dictionary[action] in ['south', 'north', 'east', 'west']:
                        moved, row, col = self._take_movement(index_action_dictionary[action], row, col)

                    # Check for collisions
                    if self.collision_sensitive_domain and moved:
                        if self.collided[taxi] == 0:
                            reward, moved, action, taxis_locations = self._check_action_for_collision(taxi,
                                                                                                      taxis_locations,
                                                                                                      row, col, moved,
                                                                                                      action, reward)

                    # Pickup
                    elif index_action_dictionary[action] == 'pickup':
                        passengers_status, reward = self._make_pickup(taxi, passengers_start_locations,
                                                                      passengers_status, taxi_location, reward)

                    # Dropoff
                    elif index_action_dictionary[action] == 'dropoff':
                        passengers_status, passengers_start_locations, reward = self._make_dropoff(taxi,
                                                                                                   passengers_start_locations,
                                                                                                   passengers_status,
                                                                                                   destinations,
                                                                                                   taxi_location,
                                                                                                   reward)

                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        reward = self.partial_closest_path_reward('turn_engine_off')
                        self.engine_status_list[taxi] = 0

                    # Standby with engine on
                    elif index_action_dictionary[action] == 'standby':
                        reward = self.partial_closest_path_reward('standby_engine_on')

                # Here we have finished checking for action for taxi-i
                # Fuel consumption
                if moved:
                    reward, fuels[taxi], taxis_locations = self._update_movement_wrt_fuel(taxi, taxis_locations,
                                                                                          row, col, reward, fuel)

                if (not moved) and action in [self.action_index_dictionary[direction] for
                                              direction in ['north', 'south', 'west', 'east']]:
                    reward = self.reward_dict['hit_wall']

                # taxi refuel
                if index_action_dictionary[action] == 'refuel':
                    reward, fuels[taxi] = self._refuel_taxi(fuel, reward, taxi, taxis_locations)

                # check if all the passengers are at their destinations
                done = all(loc == 1 for loc in passengers_status)
                self.dones[taxi_name] = done

                # check if all taxis collided
                done = all(self.collided == 1)
                self.dones[taxi_name] = self.dones[taxi_name] or done

                # check if all taxis are out of fuel
                done = fuels[taxi] == 0
                self.dones[taxi_name] = self.dones[taxi_name] or done

                rewards[taxi_name] = reward
                self.state = [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
                self.last_action = action_dict

        self.dones['__all__'] = True
        self.dones['__all__'] = all(list(self.dones.values()))

        if self.display:
            self.render()

        if self.bounded:
            total_reward = 0
            for taxi_id in action_dict.keys():
                total_reward += rewards[taxi_id]
            total_reward /= len(action_dict.keys())
            for taxi_id in action_dict.keys():
                rewards[taxi_id] = total_reward

        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self.get_observation(self.state, taxi_id)

        return obs, {taxi_id: rewards[taxi_id] for taxi_id in action_dict.keys()}, self.dones, {}

    def partial_closest_path_reward(self, basic_reward_str: str, taxi_index: int = None) -> int:
        """
        Computes the reward for a taxi and it's defined by:
        dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
        away - negative.
        other actions - basic reward from config table
        Args:
            basic_reward_str: the reward we would like to give
            taxi_index: index of the specific taxi

        Returns: updated reward

        """
        if basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
            return self.reward_dict[basic_reward_str]

        # [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
        current_state = self.state
        passengers_start_locations = current_state[2]

        taxis_locations = current_state[0]

        passengers_status = current_state[-1]
        passenger_index = passengers_status.index(taxi_index + 3)
        passenger_start_row, passenger_start_col = passengers_start_locations[passenger_index]
        taxi_current_row, taxi_currrent_col = taxis_locations[taxi_index]

        return 15 * (self.passenger_destination_l1_distance(passenger_index, passenger_start_row, passenger_start_col) -
                     self.passenger_destination_l1_distance(passenger_index, taxi_current_row, taxi_currrent_col))


class TaxiLockTaxiStartPositionTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        taxis_locations = [[2, 2]]
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = 5
        self.counter = 0

        # refuel everybody
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]

        # reset passengers
        passengers_start_location = [start for start in
                                     random.choices(self.passengers_locations, k=self.num_passengers)]
        passengers_destinations = [random.choice([x for x in self.passengers_locations if x != start])
                                   for start in passengers_start_location]

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


class TaxiDeterministicPositionTransform(TaxiTransformedEnv):
    def __init__(self, x=None, **kwargs):
        self.reward_dict = NEW_REWARD
        self.display = False
        super().__init__(domain_map=NEW_MAP)

    def set_display(self, val):
        self.display = val

    def reset(self):
        taxis_locations = [[2, 2]]
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = 5
        self.counter = 0

        # refuel everybody
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]

        # reset passengers
        passengers_start_location = [[0, 0]]
        passengers_destinations = [[3, 0]]

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

    def set_reward_dict(self, new_rewards):
        self.reward_dict = new_rewards

    def step(self, action_dict: dict) -> (dict, dict, dict, dict):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.
        Using manual reward.

        Args:
            action_dict: {taxi_name: action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """

        rewards = {}
        self.counter += 1
        if self.counter >= 90:
            self.window_size = 3

        # Main of the function, for each taxi-i act on action[i]
        for taxi_name, action_list in action_dict.items():
            # meta operations on the type of the action
            action_list = self._get_action_list(action_list)

            for action in action_list:
                taxi = self.taxis_names.index(taxi_name)
                reward = self.partial_closest_path_reward('step')  # Default reward
                moved = False  # Indicator variable for later use

                # taxi locations: [i, j]
                # fuels: int
                # passengers_start_locations and destinations: [[i, j] ... [i, j]]
                # passengers_status: [[1, 2, taxi_index+2] ... [1, 2, taxi_index+2]], 1 - delivered
                taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state

                if all(list(self.dones.values())):
                    continue

                # If taxi is collided, it can't perform a step
                if self.collided[taxi] == 1:
                    rewards[taxi_name] = self.partial_closest_path_reward('collided')
                    self.dones[taxi_name] = True
                    continue

                # If the taxi is out of fuel, it can't perform a step
                if fuels[taxi] == 0 and not self.at_valid_fuel_station(taxi, taxis_locations):
                    rewards[taxi_name] = self.partial_closest_path_reward('bad_fuel')
                    self.dones[taxi_name] = True
                    continue

                taxi_location = taxis_locations[taxi]
                row, col = taxi_location

                fuel = fuels[taxi]
                is_taxi_engine_on = self.engine_status_list[taxi]
                _, index_action_dictionary = self.get_available_actions_dictionary()

                if not is_taxi_engine_on:  # Engine is off
                    # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                    reward = self._engine_is_off_actions(index_action_dictionary[action], taxi)

                else:  # Engine is on
                    # Binding
                    if index_action_dictionary[action] == 'bind':
                        self.bounded = False
                        reward = self.partial_closest_path_reward('bind')

                    # Movement
                    if index_action_dictionary[action] in ['south', 'north', 'east', 'west']:
                        moved, row, col = self._take_movement(index_action_dictionary[action], row, col)

                    # Check for collisions
                    if self.collision_sensitive_domain and moved:
                        if self.collided[taxi] == 0:
                            reward, moved, action, taxis_locations = self._check_action_for_collision(taxi,
                                                                                                      taxis_locations,
                                                                                                      row, col, moved,
                                                                                                      action, reward)

                    # Pickup
                    elif index_action_dictionary[action] == 'pickup':
                        passengers_status, reward = self._make_pickup(taxi, passengers_start_locations,
                                                                      passengers_status, taxi_location, reward)

                    # Dropoff
                    elif index_action_dictionary[action] == 'dropoff':
                        passengers_status, passengers_start_locations, reward = self._make_dropoff(taxi,
                                                                                                   passengers_start_locations,
                                                                                                   passengers_status,
                                                                                                   destinations,
                                                                                                   taxi_location,
                                                                                                   reward)

                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        reward = self.partial_closest_path_reward('turn_engine_off')
                        self.engine_status_list[taxi] = 0

                    # Standby with engine on
                    elif index_action_dictionary[action] == 'standby':
                        reward = self.partial_closest_path_reward('standby_engine_on')

                # Here we have finished checking for action for taxi-i
                # Fuel consumption
                if moved:
                    reward, fuels[taxi], taxis_locations = self._update_movement_wrt_fuel(taxi, taxis_locations,
                                                                                          row, col, reward, fuel)

                if (not moved) and action in [self.action_index_dictionary[direction] for
                                              direction in ['north', 'south', 'west', 'east']]:
                    reward = self.reward_dict['hit_wall']

                # taxi refuel
                if index_action_dictionary[action] == 'refuel':
                    reward, fuels[taxi] = self._refuel_taxi(fuel, reward, taxi, taxis_locations)

                # check if all the passengers are at their destinations
                done = all(loc == 1 for loc in passengers_status)
                self.dones[taxi_name] = done

                # check if all taxis collided
                done = all(self.collided == 1)
                self.dones[taxi_name] = self.dones[taxi_name] or done

                # check if all taxis are out of fuel
                done = fuels[taxi] == 0
                self.dones[taxi_name] = self.dones[taxi_name] or done

                rewards[taxi_name] = reward
                self.state = [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
                self.last_action = action_dict

        self.dones['__all__'] = True
        self.dones['__all__'] = all(list(self.dones.values()))

        if self.display:
            self.render()

        if self.bounded:
            total_reward = 0
            for taxi_id in action_dict.keys():
                total_reward += rewards[taxi_id]
            total_reward /= len(action_dict.keys())
            for taxi_id in action_dict.keys():
                rewards[taxi_id] = total_reward

        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self.get_observation(self.state, taxi_id)

        return obs, {taxi_id: rewards[taxi_id] for taxi_id in action_dict.keys()}, self.dones, {}

    def partial_closest_path_reward(self, basic_reward_str: str, taxi_index: int = None) -> int:
        """
        Computes the reward for a taxi and it's defined by:
        dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
        away - negative.
        other actions - basic reward from config table
        Args:
            basic_reward_str: the reward we would like to give
            taxi_index: index of the specific taxi

        Returns: updated reward

        """
        if basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
            return self.reward_dict[basic_reward_str]

        # [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
        current_state = self.state
        passengers_start_locations = current_state[2]

        taxis_locations = current_state[0]

        passengers_status = current_state[-1]
        passenger_index = passengers_status.index(taxi_index + 3)
        passenger_start_row, passenger_start_col = passengers_start_locations[passenger_index]
        taxi_current_row, taxi_currrent_col = taxis_locations[taxi_index]

        return 15 * (self.passenger_destination_l1_distance(passenger_index, passenger_start_row, passenger_start_col) -
                     self.passenger_destination_l1_distance(passenger_index, taxi_current_row, taxi_currrent_col))


def taxi_small_map_transform(env):
    """
    set the map to be different
    """
    new_env = TaxiSmallMapTransform()
    return new_env, TaxiSmallMapTransform


def taxi_move_through_walls_transform(env):
    """
    Taxi can move through walls
    """
    new_env = TaxiNoWallsTransform()
    return new_env, TaxiNoWallsTransform


def taxi_infinite_fuel_transform(env):
    """
    Infinite fuel environment
    """
    import Environments.MultiTaxiEnv.multitaxienv.taxi_environment as taxi_env
    taxi_env.set_number_of_agents(1)
    new_env = TaxiInfiniteFuelTransform()
    return new_env, TaxiInfiniteFuelTransform


def taxi_reward_transform(env, new_rewards=None):
    """
    set the rewards in the environment
    """
    new_env = TaxiRewardTransform()
    return new_env, TaxiRewardTransform


def taxi_lock_starting_position_transform(env):
    """
    set the rewards in the environment
    """
    new_env = TaxiLockTaxiStartPositionTransform()
    return new_env, TaxiLockTaxiStartPositionTransform


def taxi_deterministic_position_transform(env):
    """
    set the rewards in the environment
    """
    new_env = TaxiDeterministicPositionTransform()
    return new_env, TaxiDeterministicPositionTransform
