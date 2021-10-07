from ray import tune
from ray.tune.schedulers import PopulationBasedTrainingReplay
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import os
import yaml
import argparse
from functools import partial

from src.utils.hyperparameter_processing import preprocess_hyperparams
from src.train import get_parser
from src.context.sampling import sample_contexts

def setup_agent(config, outdir, parser, args):
    env_wrapper = None
    env = config["env_config"]["env"]
    config["seed"] = config["env_config"]["seed"]
    seed = config["env_config"]["seed"]
    hide_context = config["env_config"]["hide_context"]
    context_args = config["env_config"]["context_args"]
    del config["env_config"]
    timesteps = 0
    config["seed"] = seed

    num_contexts = 100
    contexts = sample_contexts(
        env,
        context_args,
        num_contexts,
        default_sample_std_percentage=0.1
    )

    logger = TrialLogger(
        outdir,
        parser=parser,
        trial_setup_args=args,
        add_context_feature_names_to_logdir=False,
        init_sb3_tensorboard=False  # set to False if using SubprocVecEnv
    )

    logger.write_trial_setup()

    train_args_fname = os.path.join(logger.logdir, "trial_setup.json")
    with open(train_args_fname, 'w') as file:
        json.dump(args.__dict__, file, indent="\t")

    contexts = get_contexts(args)
    contexts_fname = os.path.join(logger.logdir, "contexts_train.json")
    with open(contexts_fname, 'w') as file:
        json.dump(contexts, file, indent="\t")

    env_logger = logger if vec_env_cls is not SubprocVecEnv else None
    from src.envs import CARLAcrobotEnv
    EnvCls = partial(
        eval(self.env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    env = make_vec_env(EnvCls, n_envs=1, wrapper_class=env_wrapper)

    model = PPO('MlpPolicy', env, **config)
    return model, timesteps, env, context_args, hide_context

def eval_model(model, eval_env):
    eval_reward = 0
    for i in range(100):
        done = False
        state = eval_env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = eval_env.step(action)
            eval_reward += reward
    return eval_reward/100

def step(model, timesteps, env, context_args, hide_context):
    model.learn(4096)
    timesteps += 4096
    num_contexts = 100
    contexts = sample_contexts(
        env,
        context_args,
        num_contexts,
        default_sample_std_percentage=0.1
    )
    env_logger = None
    from src.envs import CARLAcrobotEnv
    EnvCls = partial(
        eval(self.env),
        contexts=contexts,
        logger=env_logger,
        hide_context=hide_context,
    )
    eval_env = make_vec_env(EnvCls, n_envs=1, wrapper_class=None)
    eval_reward = self.eval_model(eval_env)
    return eval_reward, model, timesteps

def update_config(model, new_config):
    model.learning_rate = new_config["learning_rate"]
    model.gamma = new_config["gamma"]
    model.ent_coef = new_config["ent_coef"]
    model.vf_coef = new_config["vf_coef"]
    model.gae_lambda = new_config["gae_lambda"]
    model.max_grad_norm = new_config["max_grad_norm"]
    return model

def load_hps(policy_file):
    raw_policy = []
    with open(policy_file, "rt") as fp:
        for row in fp.readlines():
            parsed_row = json.loads(row)
            raw_policy.append(tuple(parsed_row))

    policy = []
    last_new_tag = None
    last_old_conf = None
    for (old_tag, new_tag, old_step, new_step, old_conf, new_conf) in reversed(raw_policy):
        if last_new_tag and old_tag != last_new_tag:
            break
        last_new_tag = new_tag
        last_old_conf = old_conf
        policy.append((new_step, new_conf))

    return last_old_conf, iter(list(reversed(policy)))

parser = argparse.ArgumentParser()
parser.add_argument(
        "--policy_path", help="Path to PBT policy")
parser.add_argument("--seed", type=int)
parser.add_argument("--env", type=str)
parser.add_argument("--hide_context", action='store_true')
parser.add_argument("--name", type=str)
parser.add_argument("--context_args", type=str)
args, _ = parser.parse_known_args()

pbt_folder = "pbt_hps"
if args.hide_context:
    pbt_folder = "pbt_hps_hidden"
outdir = f"/home/eimer/Dokumente/git/meta-gym/src/results/classic_control/{pbt_folder}/{args.env}/{args.name}"

env_config = {"seed": args.seed, "env": args.env, "hide_context": args.hide_context, "context_args": args.context_args}

config, hp_schedule = load_hps(args.policy_path)
config["env_config"] = env_config
model, timesteps, env, context_args, hide_context = setup_agent(config, outdir, parser, args)
for i in range(250):
    config = next(hp_schedule, None)
    model = update_config(mode, config)
    reward, model, timesteps = step(model, timesteps, env, context_args, hide_context)