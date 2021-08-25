# ================= Train variables ================ #
ITER_NUM = 1000
SPEAKER_LISTENER = "simple_speaker_listener"
# =================== Transforms =================== #
FUELS_TRANSFORM = "fuels_transform"
REWARD_TRANSFORM = "reward_transform"
NO_WALLS_TRANSFORM = "no_walls_transform"
WITHOUT_TRANSFORM = "Without transform"
# ================== Environments ================== #
TAXI = "taxi"
TAXI_STATES_NUM = 5000
# ----------------- Env variables ------------------ #

TRAIN = "train"
EVALUATE = "evaluate"
EPISODE_REWARD_MEAN = "episode_reward_mean"
EPISODE_REWARD_MAX = "episode_reward_max"
EPISODE_REWARD_MIN = "episode_reward_min"
EPISODE_STEP_NUM_MEAN = "episode_len_mean"
EPISODE_VARIANCE = "episode_variance"
TOTAL_EPISODE_REWARD = "total_episode_reward"
TRAINING_RESULTS = "training_results"
EVALUATION_RESULTS = "evaluation_results"
SUCCESS_RATE = "success_rate"
GOT_AN_EXPLANATION = "explanation"
ORIGINAL_ENV = "original_env"
FUEL_TRANSFORMED_ENV = "fuel"

ANTICIPATED_POLICY = {(2, 0, 0, 3, None): [1],
                      (1, 0, 0, 3, None): [1],
                      (0, 0, 0, 3, None): [4],
                      (0, 0, 4, 3, None): [0],
                      (1, 0, 4, 3, None): [2],
                      (1, 1, 4, 3, None): [2],
                      (1, 2, 4, 3, None): [0],
                      (2, 2, 4, 3, None): [5]}
