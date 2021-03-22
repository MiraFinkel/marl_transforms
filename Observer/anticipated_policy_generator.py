import numpy as np


class OptimalAgent:
    def __init__(self, env):
        self.env = env
        self.nA = len(self.env.get_available_actions_dictionary()[0])
        self.nS = self._get_state_number()
        self.policy_dict = {}

    def value_iteration(self, theta=1, discount_factor=0.99, display=True):
        """
        Value Iteration Algorithm.

        Args:
            # env: OpenAI env. env.P represents the transition probabilities of the environment.
            #     env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            #     env.nS is a number of states in the environment.
            #     env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.
            display: displays iteration num

        Returns:
            A tuple (policy, V) of the optimal policy and the optimal value function.
        """

        def one_step_lookahead(state, V):
            """
            Helper function to calculate the value for all action in a given state.

            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS

            Returns:
                A vector of length env.nA containing the expected value of each action.
            """
            A = np.zeros(self.nA)
            state = list(self.env.decode(state))
            self.env.reset()
            self.env.state = state
            for a in range(self.nA):
                taxi_name = "taxi_1"
                next_state, reward, done, _ = self.env.step({taxi_name: a})
                next_state, reward, done, prob = next_state[taxi_name][0], reward[taxi_name], done[taxi_name], 1.0
                # for prob, next_state, reward, done in env.P[state][a]:
                cur_next_state = self.env.encode(next_state[0], next_state[1], next_state[2], next_state[3],
                                                 next_state[4], next_state[5], next_state[6], next_state[7])
                #  taxi_row, taxi_col, fuel, pass_loc_x, pass_loc_y, dest_idx_x, dest_idx_y, pass_status
                A[a] += prob * (reward + discount_factor * V[cur_next_state - 1])
            return A

        V = np.zeros(self.nS)
        i = 0
        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in range(self.nS):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
                # Check if we can stop
            i += 1
            if display:
                print("Optimal agent iteration num: ", i)
            if delta < theta:
                break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([self.nS, self.nA])
        policy_dict = {}
        for s in range(self.nS):
            decoded_state = list(self.env.decode(s))
            flatten_state = decoded_state[0][0] + decoded_state[1] + decoded_state[2][0] + decoded_state[3][0] + \
                            decoded_state[4]
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0
            policy_dict[tuple(flatten_state)] = best_action
        self.policy_dict = policy_dict
        return policy_dict, policy, V

    def _get_state_number(self):
        taxi_possible_locations = self.env.num_rows * self.env.num_columns
        fuel = self.env.max_fuel[0] + 1
        passenger_possible_locations = self.env.num_rows * self.env.num_columns
        passenger_possible_destinations = len(self.env.passengers_locations)
        passengers_status = 3
        return taxi_possible_locations * fuel * passenger_possible_locations * passenger_possible_destinations * passengers_status

    def compute_action(self, state):
        return self.policy_dict[tuple(state[0])]


# helper function for flattening irregular nested tuples
def mixed_flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__"):
            result.extend(mixed_flatten(el))
        else:
            result.append(el)
    return result


# helper function for making a list of coordinates of interest
# make list of valid coords in environment within dist of the given loc
def get_nearby_coords(env, loc, dist):  # option for later: add an option to change definition of distance
    max_rows = env.num_rows - 1
    max_cols = env.num_columns - 1
    (x, y) = loc
    result = []
    for i in range(x - dist, x + dist + 1):
        for j in range(y - dist, y + dist + 1):
            if 0 <= i <= max_rows and 0 <= j <= max_cols:
                result.append((i, j))
    return result


def sample_anticipated_policy(optimal_agent, env, num_states_in_partial_policy):
    passenger_origin = get_possible_passenger_origins(env)
    passenger_destination = get_possible_passenger_destinations(env)

    policy_dict = optimal_agent.policy_dict
    optimal_policy = {}
    for k in policy_dict.keys():
        if is_interesting_state(k, passenger_origin, passenger_destination):
            optimal_policy[k] = policy_dict[k]

    sampled_states_flat = np.random.choice(len(optimal_policy), size=num_states_in_partial_policy,
                                           replace=False)  # get flat indices of sampled states

    partial_sampled_policy = {}
    for i, item in enumerate(optimal_policy.items()):
        if i in sampled_states_flat:
            list(item[0])[2] = None
            partial_sampled_policy[tuple(item[0])] = item[1]

    return partial_sampled_policy


def is_interesting_state(state, passenger_origins, passenger_destinations):
    taxi_location = [state[0], state[1]]
    fuel_level = state[2]
    passenger_location = [state[3], state[4]]
    passenger_destination = [state[5], state[6]]
    passenger_status = state[6]

    fuel_is_full = (fuel_level == 100)
    taxi_in_interesting_location = (
                (taxi_location[0] == passenger_location[0] and taxi_location[1] == passenger_location[1]) or (
                    taxi_location[0] == passenger_destination[0] and taxi_location[1] == passenger_destination[1]))
    passenger_in_interesting_location = passenger_location in passenger_origins
    valid_passenger_destination = ((passenger_destination[0] == passenger_location[0]) and (
            passenger_destination[1] == passenger_location[1]) and passenger_status > 2) or (
                                          passenger_destination[0] != passenger_location[0]) or (
                                          passenger_destination[1] != passenger_location[1])

    if fuel_is_full and taxi_in_interesting_location and passenger_in_interesting_location and valid_passenger_destination:
        return True
    return False


def get_possible_passenger_origins(env):
    return env.passengers_locations


def get_possible_passenger_destinations(env):
    return env.passengers_locations
