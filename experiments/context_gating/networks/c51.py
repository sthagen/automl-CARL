import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

from ..networks.common import context_gating_func


def q_func(cfg, env):
    def q(S, A, is_training):
        if cfg.carl.dict_observation_space and not cfg.carl.hide_context:
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                )
            )
            context_gating = context_gating_func(cfg)
            X = jnp.concatenate((S["state"], A), axis=-1)
            x = state_seq(X)
            if cfg.q_context:
                x = context_gating(x, S)
            q_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.num_atoms, w_init=jnp.zeros),
                )
            )
            x = q_seq(x)
        else:
            # TODO(carolin): implement Nature DQN conv encoder (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py#L113)
            X = jnp.concatenate((S, A), axis=-1)
            state_seq = hk.Sequential(
                (
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.width),
                    jax.nn.relu,
                    hk.Linear(cfg.network.num_atoms, w_init=jnp.zeros),
                )
            )
            x = state_seq(X)
        return {"logits": x}

    return q
