import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)
import numpy as np
from functools import partial
import importlib
from rich import print
from pathlib import Path
import time
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig
import coax

import carl.envs as envs
import jax
import numpy as onp
import wandb

from experiments.context_gating.algorithms.td3 import td3
from experiments.context_gating.algorithms.sac import sac
from experiments.context_gating.utils import set_seed_everywhere
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, CategoricalHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

import carl.envs
from experiments.common.train.utils import make_carl_env
from experiments.common.utils.search_space_encoding import search_space_to_config_space
from experiments.attack_on_agents.agent_creation import make_agent

from carl.context.sampling import sample_contexts

import pdb

def get_default_context(env_name: str) -> Dict[Any, Any]:
    env_cls = getattr(carl.envs, env_name)
    env_module = importlib.import_module(env_cls.__module__)
    context_def = getattr(env_module, "DEFAULT_CONTEXT")

    default_hp = {}

    for context_name in context_def:
        default_hp[context_name] = False

    default_hp['g'] = True

    return default_hp

def context_features_to_configuration_space(env_name: str) -> ConfigurationSpace:
    env_cls = getattr(carl.envs, env_name)
    env_module = importlib.import_module(env_cls.__module__)
    #context_def = getattr(env_module, "DEFAULT_CONTEXT")
    context_bounds = getattr(env_module, "CONTEXT_BOUNDS")

    first = True
    hyperparameters = []
    for cf_name, _ in context_bounds.items():
        

        if 'initial' not in cf_name:
            print(cf_name)
            pdb.set_trace()
            hp = CategoricalHyperparameter(
                cf_name,
                choices = [False, True],
                default_value = first 
            )

            first = False
            
            hyperparameters.append(hp)
    
    
    configuration_space = ConfigurationSpace()
    configuration_space.add_hyperparameters(hyperparameters=hyperparameters)

    return configuration_space

def eval_agent(
    config_smac: Configuration,             # Configuration to handle contexts 
    cfg: DictConfig,                        
    file_id: Optional[str] = None   ) -> float:
    
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    wandb.init(
        mode="offline" if cfg.debug else None,
        project="carl",
        entity="tnt",
        group=cfg.group,
        dir=os.getcwd(),
        config=dict_cfg,
    )
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    wandb.config.update({"command": command, "slurm_id": slurm_id})
    set_seed_everywhere(cfg.seed)

    EnvCls = partial(getattr(envs, cfg.env), **cfg.carl)

    context_args = []    
    
    
    # TODO check if this works
    for k in config_smac:        
        if config_smac[k] is True:
            context_args.append(k)

    print('context_args', context_args)
    
    # Train contexts sampled from 
    # SMAC optimization
    train_contexts = sample_contexts(
                            env_name = cfg.env,  
                            context_feature_args= context_args, 
                            **cfg.train_contexts
                        )

    # print('train_contexts', train_contexts[0])

    # Test contexts for extrapolation
    eval_contexts =  sample_contexts(
                        env_name = cfg.env,
                        context_feature_args= context_args, 
                        **cfg.eval_contexts
                    )#

    # print('eval_contexts', eval_contexts[0])
    # pdb.set_trace()



    # Create environments
    env = EnvCls(contexts=train_contexts, context_encoder=None)
    eval_env = EnvCls(contexts=eval_contexts, context_encoder=None)

    env = coax.wrappers.TrainMonitor(env, name=cfg.algorithm)
    key = jax.random.PRNGKey(cfg.seed)

    if cfg.state_context and cfg.carl.dict_observation_space:
        key, subkey = jax.random.split(key)
        context_state_indices = jax.random.choice(
            subkey,
            onp.prod(env.observation_space.spaces["state"].low.shape),
            shape=env.observation_space.spaces["context"].shape,
            replace=True,
        )
        print(f"Using state features {context_state_indices} as context")
    else:
        context_state_indices = None
    cfg.context_state_indices = context_state_indices

    avg_return = sac(cfg, env, eval_env)

    run_data = np.array({
        "train_context": context_args,
        "returns": avg_return,
    })


    if file_id is None:
        file_id = time.time_ns()
    fp = Path(f"./eval_data/eval_data_{file_id}.npz")
    fp.parent.mkdir(exist_ok=True, parents=True)
    run_data.dump(fp)

    # TODO normalize reward?
    return avg_return


def create_tae_runner(cfg: DictConfig) -> callable:
    return partial(
        eval_agent,
        cfg=cfg,
        file_id= cfg.env + '_' + str(time.time_ns()) 
    )


@hydra.main("./configs", "base")
def main(cfg: DictConfig):
    print(cfg)
    configuration_space = context_features_to_configuration_space(env_name=cfg.env)
    configuration_space.seed(cfg.seed)
    #print(configuration_space)

    scenario = Scenario({
            "run_obj": "quality",           # we optimize quality (alternatively runtime)
            "runcount-limit": cfg.budget,   # max. number of function evaluations
            "cs": configuration_space,      # configuration space
            "deterministic": True,
        }
    )

    tae_runner = create_tae_runner(cfg=cfg)
    # context_default = get_default_context(env_name=cfg.env)
    
    # performance_default = tae_runner(configuration_space.get_default_configuration(), file_id="default")

    # print(performance_default)
    # pdb.set_trace()

    smac = SMAC4BB(
        scenario=scenario,
        model_type="gp",
        rng=np.random.RandomState(cfg.seed),
        acquisition_function=EI,  # or others like PI, LCB as acquisition functions
        tae_runner=tae_runner,
        initial_design_kwargs={"init_budget": 2}
    )

    smac.optimize()

    return configuration_space


if __name__ == "__main__":
    main()
    #print(context_features_to_configuration_space(env_name="CARLPendulumEnv"))
