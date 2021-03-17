import numpy as np


class OptimalAgent:
    def __init__(self, env):
        self.env = env
        self.nA = len(self.env.get_available_actions_dictionary()[0])
        # self.nS = env.num_states * 4  # TODO - fix the num of states in the taxi domain?
        self.nS = self._get_state_number()

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
        for s in range(self.nS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0

        return policy, V

    def _get_state_number(self):
        taxi_possible_locations = self.env.num_rows * self.env.num_columns
        fuel = self.env.max_fuel[0] + 1
        passenger_possible_locations = self.env.num_rows * self.env.num_columns
        passenger_possible_destinations = len(self.env.passengers_locations)
        passengers_status = 3
        return taxi_possible_locations * fuel * passenger_possible_locations * passenger_possible_destinations * passengers_status
