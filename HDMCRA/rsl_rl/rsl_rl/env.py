"""Minimal VecEnv interface for compatibility with legged_gym task registry."""


class VecEnv:
    """Abstract base class for vectorized environments."""

    def get_observations(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError
