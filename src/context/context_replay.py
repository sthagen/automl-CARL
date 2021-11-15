import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

class ContextReplayBuffer(ReplayBuffer):
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
        super(ContextReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        self.contexts = np.zeros((self.buffer_size, self.n_envs, context_dim), dtype=np.float32)
        if explicit_context:
            self.obs_shape = [self.obs_shape[0] - context_dim]
            self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                         dtype=observation_space.dtype)

            if optimize_memory_usage:
                # `observations` contains also the next observation
                self.next_observations = None
            else:
                self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                                  dtype=observation_space.dtype)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            context: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.contexts[self.pos] = np.array(context).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

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
        sample_context = np.random.choice(np.unique(self.contexts))
        eligible_indices = [i for i in range(len(self.contexts)) if self.contexts[i]==sample_context]
        batch_inds = np.random.choice(eligible_indices, size=batch_size)
        return self._get_samples(batch_inds, env=env)


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
        self.priorities = self.contexts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            context: np.ndarray,
            td_errors: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, context, infos)
        self.priorities[self.pos] = np.array(td_errors).copy() * self.alpha + self.epsilon

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        contexts = np.unique(self.contexts)
        context_weights = []
        for c in contexts:
            context_indices = [i for i in range(self.pos) if self.contexts[i]==c]
            context_priorities = [self.priorities[i] for i in context_indices]
            context_weights.append(max(context_priorities))

        sample_context = np.random.choice(contexts, p=context_weights)
        eligible_indices = [i for i in range(len(self.contexts)) if self.contexts[i] == sample_context]
        batch_inds = np.random.choice(eligible_indices, size=batch_size)
        return self._get_samples(batch_inds, env=env)


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

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        available_contexts = np.unique(self.contexts)
        if len(available_contexts) >= batch_size:
            sampled_contexts = np.random.choice(available_contexts, size=batch_size)
        else:
            sampled_contexts = np.concatenate((available_contexts,np.random.choice(available_contexts, size=batch_size-len(available_contexts))))

        batch_inds = []
        for c in sampled_contexts:
            context_indices = [i for i in range(self.pos) if self.contexts[i]==c]
            batch_inds.append(np.random.choice(context_indices))
        return self._get_samples(batch_inds, env=env)