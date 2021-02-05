from Transforms.transforms import *

# ===================== Agents ===================== #
A2C = "a2c"
A3C = "a3c"
BC = "bc"
DQN = "dqn"
RAINBOW = "rainbow"
APEX_DQN = "apex_dqn"
IMPALA = "impala"
MARWIL = "marwil"
PG = "pg"
PPO = "ppo"
APPO = "appo"
SAC = "sac"
LIN_UCB = "lin_usb"
LIN_TS = "lin_ts"



AGENT = PG
# ================= Train variables ================ #
ITER_NUM = 2000
NUM_GPUS = 0
NUM_WORKERS = 1
WITH_DEBUG = True
# =================== Transforms =================== #
WITH_TRANSFORM = True
TRANSFORM = DimReductionMultiAgents
DIM_REDUCTION_IDX = 0
NO_TAXI_LOC_TRANSFORM = "no taxi loc transform"
NO_FUELS_TRANSFORM = "no fuels transform"
WITHOUT_TRANSFORM = "Without transform"
NO_PASS_START_LOC_TRANSFORM = "no pass start loc transform"
NO_PASS_DEST_LOC_TRANSFORM = "no pass dest transform"
NO_PASS_STATUS_LOC_TRANSFORM = "no pass status transform"
# ================== Environments ================== #
ENV = TransformEnvironment
TAXI = "taxi"
ENV_NAME = TAXI
# ----------------- Env variables ------------------ #
NUM_TAXI = 2
TAXI1_GAMMA = 0.85
TAXI2_GAMMA = 0.95
