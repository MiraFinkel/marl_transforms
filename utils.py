from constants import *
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
import ray.rllib.agents.pg as pg
import ray.rllib.agents.a3c as a3c


def get_env(env_name, with_transform=False, transform_idx=0):
    if not with_transform:
        if env_name == TAXI:
            return TaxiEnv(), TaxiEnv
    else:
        transform_env = TransformEnvironment()
        transform_env._mapping_class.set_reduction_idx(transform_idx)
        return transform_env, TransformEnvironment


def get_agent(agent_name, config, env_to_agent):
    if agent_name == PG:
        return pg.PGTrainer(config=config, env=env_to_agent)
    elif agent_name == A3C:
        return a3c.A3CTrainer(config=config, env=env_to_agent)
    elif agent_name == A2C:
        return a3c.A2CTrainer(config=config, env=env_to_agent)


def get_taxi_config(env, number_of_taxis):
    config = {}
    if number_of_taxis == 2:
        config = {'multiagent': {'policies': {'taxi_1': (None, env.obs_space, env.action_space, {'gamma': TAXI1_GAMMA}),
                                              'taxi_2': (None, env.obs_space, env.action_space, {'gamma': TAXI2_GAMMA})
                                              },
                                 "policy_mapping_fn": lambda taxi_id: taxi_id},
                  "num_gpus": NUM_GPUS,
                  "num_workers": NUM_WORKERS}

    return config
