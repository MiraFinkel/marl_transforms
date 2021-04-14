from Transforms.transforms import *



# ================= Train variables ================ #
ITER_NUM = 1000
SPEAKER_LISTENER = "simple_speaker_listener"
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

ENV_NAME = TAXI
# ----------------- Env variables ------------------ #

TRAIN = "train"
EVALUATE = "evaluate"
