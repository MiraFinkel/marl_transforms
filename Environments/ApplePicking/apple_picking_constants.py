SOUTH, NORTH, EAST, WEST, PICKUP = 0, 1, 2, 3, 4
ACTIONS = [SOUTH, NORTH, EAST, WEST, PICKUP]

STEP_REWARD, PICKUP_REWARD, BAD_PICKUP_REWARD = "step", "good_pickup", "bad_pickup"
REWARD_DICT = {STEP_REWARD: -1, PICKUP_REWARD: 10, BAD_PICKUP_REWARD: -10}