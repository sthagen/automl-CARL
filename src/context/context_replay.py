import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

from gym import spaces
from typing import List, Dict, Union, Optional, Any

class ContextReplayBuffer(ReplayBuffer):
    def __init__(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        context_dim: int = None,
        explicit_context: bool = True,
        context_feature = None
    ):
        super(ContextReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.context_dim = context_dim
        if not self.context_dim:
            raise ValueError("Please add the dimension of your context features")
        self.contexts = np.zeros((self.buffer_size, self.n_envs, context_dim), dtype=np.float32)
        self.observation_base = np.zeros((self.buffer_size//100, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.action_base = np.zeros((self.buffer_size//100, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.reward_base = np.zeros((self.buffer_size//100, self.n_envs), dtype=np.float32)
        self.done_base = np.zeros((self.buffer_size//100, self.n_envs), dtype=np.float32)
        self.observations = dict()
        self.actions = dict()
        self.rewards = dict()
        self.dones = dict()
        self.next_observations = dict()
        self.contexts = []
        self.pos = dict()
        self.context_feature = context_feature

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for i in range(len(infos)):
            cv = infos[i]["context"][self.context_feature]
            if not cv in self.observations.keys():
                self.observations[cv] = self.observation_base.copy()
                self.next_observations[cv] = self.observation_base.copy()
                self.actions[cv] = self.action_base.copy()
                self.rewards[cv] = self.reward_base.copy()
                self.dones[cv] = self.done_base.copy()
                self.contexts.append(cv)
                self.pos[cv] = 0
            self.observations[cv][self.pos[cv]] = np.array(next_obs[i]).copy()
            self.next_observations[cv][self.pos[cv]] = np.array(next_obs[i]).copy()
            self.actions[cv][self.pos[cv]] = np.array(action[i]).copy()
            self.rewards[cv][self.pos[cv]] = np.array(reward[i]).copy()
            self.dones[cv][self.pos[cv]] = np.array(done[i]).copy()
            self.pos[cv] += 1
            if self.pos[cv] == self.buffer_size//100:
                self.full = True
                self.pos[cv] = 0
        #self.observations[self.pos] = np.array(obs).copy()
        #if self.optimize_memory_usage:
        #    self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        #else:
        #    self.next_observations[self.pos] = np.array(next_obs).copy()

        #self.actions[self.pos] = np.array(action).copy()
        #self.rewards[self.pos] = np.array(reward).copy()
        #self.dones[self.pos] = np.array(done).copy()
        #self.contexts[self.pos] = [np.round(i["context"], decimals=3).copy() for i in infos]

        #if self.handle_timeout_termination:
        #    self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        #self.pos += 1
        #if self.pos == self.buffer_size:
         #   self.full = True
          #  self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        unique_contexts = self.contexts#np.unique(self.contexts, axis=1)
        #unique_contexts.reshape((unique_contexts.shape[0]*unique_contexts.shape[1], unique_contexts.shape[2]))
        sample_context = np.random.choice(unique_contexts)
        eligible_indices = np.arange(len(np.nonzero(self.observations[sample_context])))#np.where(np.array_equal(self.contexts, sample_context))
        #eligible_indices = [i for i in range(len(self.contexts)) if np.array_equal(self.contexts[i][0], sample_context)]
        batch_inds = np.random.choice(eligible_indices, size=batch_size)
        return self._get_samples(batch_inds, sample_context, env=env)

    def _get_samples(self, batch_inds: np.ndarray, context, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[context][(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[context][batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[context][batch_inds, env_indices, :], env),
            self.actions[context][batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[context][batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[context][batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class PrioritizedContextReplayBuffer(ContextReplayBuffer):
    def __init(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        context_dim: int,
        explicit_context: bool = True,
        device: Union[th.device, str] = "cpu",
        alpha: float = 0.5,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ContextReplayBuffer, self).__init__(buffer_size, observation_space, action_space, context_dim, explicit_context, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.alpha = alpha
        self.epsilon = 0.01
        self.priority_base = self.reward_base.copy()
        self.priorities = dict()

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            td_errors: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        for i in range(len(infos)):
            cv = infos[i]["context"][self.context_feature]
            if not cv in self.observations.keys():
                self.priorities[cv] = self.priority_base.copy()
                self.priorities[cv][0] = np.array(td_errors.copy()) * self.alpha + self.epsilon
            else:
                self.priorities[cv][self.pos[cv]] = np.array(td_errors.copy()) * self.alpha + self.epsilon
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        contexts = self.contexts
        context_weights = [max(contexts[k]) for k in contexts.keys()]
        #for c in contexts:
        #    context_indices = [i for i in range(self.pos) if self.contexts[i]==c]
        #    context_priorities = [self.priorities[i] for i in context_indices]
        #    context_weights.append(max(context_priorities))

        sample_context = np.random.choice(contexts, p=context_weights)
        #eligible_indices = [i for i in range(len(self.contexts)) if self.contexts[i] == sample_context]
        eligible_indices = np.arange(len(np.nonzero(self.observations[sample_context])))
        batch_inds = np.random.choice(eligible_indices, size=batch_size)
        return self._get_samples(batch_inds, sample_context, env=env)


class ContextDiversificationReplayBuffer(ContextReplayBuffer):
    def __init(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        context_dim: int,
        explicit_context: bool = True,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ContextReplayBuffer, self).__init__(buffer_size, observation_space, action_space, context_dim, explicit_context, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

    #TODO: id list
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        #available_contexts = np.unique(self.contexts, axis=0)
        #available_contexts = np.squeeze(available_contexts)

        if len(self.contexts) >= batch_size:
            indices = np.random.choice(len(self.contexts), size=batch_size)[0]
            sampled_contexts = self.contexts[indices]#np.random.choice(len(available_contexts), size=batch_size)]
        else:
            sampled_contexts = self.contexts.copy()
            rest_size = batch_size-len(self.contexts)
            for i in range(rest_size):
                sampled_contexts.append(np.random.choice(self.contexts))
            #sampled_contexts = np.concatenate((self.contexts,self.contexts[np.random.choice(np.arange(len(self.contexts)), size=batch_size-len(self.contexts))][0]))
        #sampled_contexts = np.squeeze(sampled_contexts)
        samples = {"observations": th.tensor(th.empty(0)), "actions": th.tensor(th.empty(0)), "rewards": th.tensor(th.empty(0)), "dones": th.tensor(th.empty(0)), "next_observations": th.tensor(th.empty(0))}
        for c in sampled_contexts:
            #context_indices = [i for i in range(self.pos) if np.array_equal(self.contexts[i][0], c)]
            #if len(context_indices) < 1:
            #    batch_inds.append(np.random.choice(np.arange(len(self.contexts))))
            #else:
            #    batch_inds.append(np.random.choice(context_indices))
            index = np.random.choice(np.arange(len(self.observations[c])))
            c_sample = self._get_samples([index], c, env=env)
            samples["observations"] = th.cat((samples["observations"], c_sample.observations))
            samples["rewards"] = th.cat((samples["rewards"], c_sample.rewards))
            samples["actions"] = th.cat((samples["actions"], c_sample.actions))
            samples["dones"] = th.cat((samples["dones"], c_sample.dones))
            samples["next_observations"] = th.cat((samples["next_observations"], c_sample.next_observations))
        data = (
            samples["observations"],
            samples["actions"],
            samples["next_observations"],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            samples["dones"], #* (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            samples["rewards"],
        )
        return ReplayBufferSamples(**samples)
