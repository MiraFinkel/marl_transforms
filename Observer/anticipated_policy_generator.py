import numpy as np


class OptimalAgent:
    def __init__(self, env):
        self.env = env
        self.nA = len(self.env.get_available_actions_dictionary()[0])
        # self.nS = env.num_states * 4  # TODO - fix the num of states in the taxi domain?
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
                A[a] += prob * (reward + discount_factor * V[cur_next_state])
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

    passenger_origin_nearby_coords = [get_nearby_coords(env, passenger_origin[i], 0)[0] for i in range(len(passenger_origin))]
    passenger_destination_nearby_coords = [tuple(get_nearby_coords(env, passenger_destination[i], 0))[0] for i in range(len(passenger_destination))]

    taxi_locations_of_interest = list(set(passenger_origin_nearby_coords + passenger_destination_nearby_coords))
    taxi_locations_of_interest = [list(loc) for loc in taxi_locations_of_interest]

    # condensing passenger locations
    state_shape_fixed_fuel = [len(taxi_locations_of_interest), 1, len(env.passengers_locations),
                              len(env.passengers_locations), 3]
    fuel_index = 1  # we fixed the fuel index, which is index 2 (third item) in the state
    num_states_fixed_fuel = np.prod(state_shape_fixed_fuel)

    # num_states_in_partial_policy = 10  # sample ten states for now as an example

    sampled_states_flat = np.random.choice(num_states_fixed_fuel, size=num_states_in_partial_policy,
                                           replace=False)  # get flat indices of sampled states
    sampled_states_unraveled = np.array(
        np.unravel_index(sampled_states_flat, state_shape_fixed_fuel)).T  # convert flat indices into state tuples
    sampled_states_unraveled[:, fuel_index] = 100  # set fuel for all sampled states to full

    partial_sampled_policy = {}
    # convert destination number into coordinates and make dictionary
    for s_raw in sampled_states_unraveled:
        s = list(s_raw)
        # convert destination and location number into coordinates
        s[2] = env.passengers_locations[s[2]]
        s[3] = env.passengers_locations[s[3]]
        s[0] = taxi_locations_of_interest[s[0]]
        s = mixed_flatten(s)
        # get optimal action
        action = optimal_agent.compute_action([s])
        # set fuel to None
        s[2] = None
        partial_sampled_policy[tuple(s)] = action

    return partial_sampled_policy


def get_possible_passenger_origins(env):
    return env.passengers_locations


def get_possible_passenger_destinations(env):
    return env.passengers_locations
