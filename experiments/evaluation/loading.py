from typing import Union
from pathlib import Path
from omegaconf import DictConfig
import coax
import warnings
import importlib


def load_func_dict(path: Union[str, Path]):
    func_dict = coax.utils.load(filepath=path)
    return func_dict


def load_policy(env, cfg: DictConfig, weights_path: Union[str, Path]):
    func_dict = load_func_dict(weights_path)
    # module = importlib.import_module(f"experiments.context_gating.networks.{cfg.algorithm}")
    # func_pi = module.pi_func(cfg, env)
    # policy = coax.Policy(
    #     func_pi,
    #     env,
    #     random_seed=cfg.seed,
    # )
    # policy.params = func_dict["pi"].params
    # policy.function_state = func_dict["pi"].function_state

    if cfg.algorithm == "c51":
        from experiments.context_gating.networks.c51 import q_func
        if type(func_dict["q"]) == coax.StochasticQ:
            q = func_dict["q"]
        else:
            func_q = q_func(cfg, env)
            # main function approximators
            q = coax.StochasticQ(
                func_q,
                env,
                value_range=(cfg.q_min_value, cfg.q_max_value),
                num_bins=cfg.network.num_atoms,
            )
            q.params = func_dict["q"]["params"]
            q.function_state = func_dict["q"]["function_state"]
        policy = coax.BoltzmannPolicy(q, temperature=cfg.pi_temperature)
    elif cfg.algorithm == "td3":
        from experiments.context_gating.networks.td3 import pi_func, q_func
        func_pi = pi_func(cfg, env)
        policy = coax.Policy(func_pi, env, random_seed=cfg.seed)
        policy.params = func_dict["pi"]["params"]
        policy.function_state = func_dict["pi"]["function_state"]
    elif cfg.algorithm == "sac":
        if type(func_dict["pi"]) == coax.Policy:
            policy = func_dict["pi"]
        else:
            from experiments.context_gating.networks.sac import pi_func, q_func
            from experiments.context_gating.algorithms.proba_dists.squashed_normal import (
                SquashedNormalDist,
            )
            func_pi = pi_func(cfg, env)
            func_q = q_func(cfg, env)

            policy = coax.Policy(
                func_pi,
                env,
                random_seed=cfg.seed,
                proba_dist=SquashedNormalDist(
                    env.action_space,
                    clip_logvar=(-10.0, 4.0),
                ),
            )
            policy.params = func_dict["pi"]["params"]
            policy.function_state = func_dict["pi"]["function_state"]    
    else:
        raise NotImplementedError(f"Adjust loading for {cfg.algorithm}.")
    
    return policy


from rich import print as printr
from rich.progress import track
import pickle
from pathlib import Path
import numpy as np
import ast
import pandas as pd
from tqdm import tqdm
from wandb.sdk.wandb_helper import parse_config
from experiments.common.utils.json_utils import lazy_json_load
from experiments.evaluation.run_evaluation import find_multirun_paths
import sys
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, Optional
from pathlib import Path
from experiments.evaluation.utils import recover_traincfg_from_wandb
from multiprocessing import Pool
from functools import partial


def load_wandb_table(fn: str | Path) -> pd.DataFrame:
    data = lazy_json_load(fn)
    data = pd.DataFrame(data=np.array(data["data"]), columns=data["columns"])
    return data

fn_config = ".hydra/config.yaml"
fn_wbsummary = "wandb/latest-run/files/wandb-summary.json"
fn_wbconfig = "wandb/latest-run/files/config.yaml"


def load_from_path(p, is_optgap_exp: bool = False):
    p = Path(p)
    fn_cfg = p / fn_config
    fn_wbsum = p / fn_wbsummary
    fn_wbcfg = p / fn_wbconfig
    # If there is no run metadata, leave
    if not fn_wbcfg.is_file() or not fn_wbsum.is_file() or not fn_cfg.is_file():
        return None

    # Load eval config
    cfg = OmegaConf.load(fn_cfg)

    # Load train config
    traincfg = recover_traincfg_from_wandb(fn_wbcfg)

    # Load summary
    summary = lazy_json_load(fn_wbsum)

    if "average_return" in summary:
        average_return = summary["average_return"]
    else:
        average_return = None

    # If the evaluation was not successful, leave
    if average_return is None:
        return None

    # Load return per context id
    path_to_table = fn_wbsum.parent / summary["return_per_context_table"]["path"]
    return_per_context = load_wandb_table(path_to_table)

    # Load eval contexts
    contexts_path = fn_wbsum.parent / summary["evalpost/contexts"]["path"]
    contexts = load_wandb_table(contexts_path)

    # Add context info to return per context id

    context_ids = return_per_context["context_id"].apply(int).to_list()
    contexts_to_table = pd.DataFrame([contexts.iloc[cidx] for cidx in context_ids])
    for col in contexts_to_table.columns:
        return_per_context[col] = contexts_to_table[col].to_numpy()
    n = len(return_per_context)
    # return_per_context["mode"] = [mode] * n
    # return_per_context["distribution_type"] = [distribution_type] * n
    # return_per_context["average_return"] = [average_return] * n

    # Get metadata
    visibility = traincfg.wandb.group
    seed = traincfg.seed
    n_contexts = traincfg.context_sampler.n_samples
    return_per_context["seed"] = seed
    return_per_context["visibility"] = visibility
    return_per_context["n_contexts"] = n_contexts

    if cfg.get("contexts_path", None):
        contexts = lazy_json_load(cfg.contexts_path)
        context_id = int(list(contexts.keys())[0])
        context_id = list(contexts.values())
        if len(context_id) == len(return_per_context):
            return_per_context["context_id"] = context_id
        else:
            warnings.warn("Context IDs not updated via the contexts_path. Mismatched length.")
    elif is_optgap_exp:
        # If it is the optimality gap experiment the train contexts are split into single context
        # files.
        contexts_train_path = traincfg.contexts_train_path
        context_id = int(Path(contexts_train_path).stem.split("_")[-1])  # --> context_{seed}_{id}.json --> context_1_1.json
        return_per_context["context_id"] = context_id

    return_per_context["path"] = str(p)
 
    reps = []
    new_df = []
    for gid, gdf in return_per_context.groupby("context_id"):
        _reps = np.arange(0, len(gdf))
        gdf["rep"] = _reps
        new_df.append(gdf)
    return_per_context = pd.concat(new_df).reset_index(drop=True)
    # return_per_context["rep"] = return_per_context.groupby(["context_id"]).apply(lambda x: )
    return return_per_context


def load_from_path_eval(p, **kwargs):
    replacements_bounds = {
        "[0.9, 1.1]": 0.1,
        "[0.75, 1.25]": 0.25,
        "[0.5, 1.5]": 0.5,
        "None": 0,
    }
    key_interval = r"$\Delta_{rel}$"
    key_visibility = "visibility"
    key_performance = "return"
    key_cf = "context_sampler.context_feature_names"

    p = Path(p)
    fn_cfg = p / fn_config
    fn_wbsum = p / fn_wbsummary
    fn_wbcfg = p / fn_wbconfig
    if not fn_wbcfg.is_file() or not fn_wbsum.is_file() or not fn_cfg.is_file():
        return None
    cfg = OmegaConf.load(fn_cfg)
    traincfg = recover_traincfg_from_wandb(fn_wbcfg)
    summary = lazy_json_load(fn_wbsum)

    if "average_return" in summary:
        average_return = summary["average_return"]
    else:
        average_return = None

    if average_return is None:
        return None

    entry = {
        "average_return": average_return,
    }

    path_to_table = fn_wbsum.parent / summary["return_per_context_table"]["path"]
    return_per_context = load_wandb_table(path_to_table)

    # Sort by context id so we can add repetition info
    return_per_context.sort_values(by="context_id", inplace=True)
    n_contexts = return_per_context["context_id"].nunique()
    n_reps = len(return_per_context) // n_contexts
    assert n_contexts * n_reps == len(return_per_context)
    repetitions = np.concatenate([np.arange(0, n_reps)] * n_contexts)
    return_per_context["rep"] = repetitions

    contexts_path = fn_wbsum.parent / summary["evalpost/contexts"]["path"]
    contexts = load_wandb_table(contexts_path)

    visibility = traincfg.wandb.group

    context_ids = return_per_context["context_id"].apply(int).to_list()
    contexts_to_table = pd.DataFrame([contexts.iloc[cidx] for cidx in context_ids])
    for col in contexts_to_table.columns:
        return_per_context[col] = contexts_to_table[col].to_numpy()
    n = len(return_per_context)
    # return_per_context["average_return"] = [average_return] * n
    return_per_context[key_visibility] = visibility
    return_per_context[key_interval] = replacements_bounds[str(OmegaConf.select(traincfg, "context_sampler.uniform_bounds_rel", default=None))]
    return_per_context["context_sampler.context_feature_names"] = str(traincfg.context_sampler.context_feature_names)
    return_per_context["seed"] = traincfg.seed
    return_per_context["algorithm"] = traincfg.algorithm

    if traincfg.experiment.startswith("kirk_ep"):
        cptp = traincfg.contexts_train_path
        # ".../${env}/contexts_A.json"
        mode = Path(cptp).stem.split("_")[-1]
        return_per_context["mode"] = mode

    if cfg.get("contexts_path", None):
        contexts = lazy_json_load(cfg.contexts_path)
        # context_id = int(list(contexts.keys())[0])
        # context_id = list(contexts.values())
        context_ids_new = list(contexts.keys())
        context_ids = return_per_context["context_id"].apply(int).to_list()
        if len(context_ids_new) == len(return_per_context):
            return_per_context["context_id"] = context_ids_new
        else:
            ids = [context_ids_new[k] for k in context_ids]
            return_per_context["context_id"] = ids
            # warnings.warn("Context IDs not updated via the contexts_path. Mismatched length.")
    return return_per_context


def load(folder_eval: str, rpc_fn: str | Path, reload_rpc: bool = False, is_optgap_exp: bool = False, load_from_path: callable = load_from_path):
    rpc_fn = Path(rpc_fn)
    if not rpc_fn.is_file():
        reload_rpc = True

    paths = find_multirun_paths(result_dir=folder_eval)

    load_from_path_partial = partial(load_from_path, is_optgap_exp=is_optgap_exp)

    if reload_rpc:
        with Pool(1) as pool:
            rpc_list = pool.map(load_from_path_partial, paths )
        rpc_list = [r for r in rpc_list if r is not None]
        df_rpc = pd.concat(rpc_list)
        df_rpc.to_csv(rpc_fn, index=False)
    else:
        df_rpc = pd.read_csv(rpc_fn)
    
    return df_rpc


if __name__ == "__main__":
    folder = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-11-07/20-45-00"
    reload = True
    path = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/context_efficiency/CARLPendulumEnv/eval/on_test/0"
    path = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/runs/optimality_gap/CARLCartPoleEnv/eval_oracle/10"
    df = load_from_path(path, is_optgap_exp=True)  # folder_eval=folder, rpc_fn=f"tmp/rpc_context_efficiency_{0}.csv", reload_rpc=reload)
    print(df["context_id"].unique())
