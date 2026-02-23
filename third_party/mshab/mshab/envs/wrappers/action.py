import gymnasium as gym


class FetchActionWrapper(gym.ActionWrapper):
    def __init__(
        self, env, stationary_base=False, stationary_torso=False, stationary_head=True
    ):
        self._stationary_base = stationary_base
        self._stationary_torso = stationary_torso
        self._stationary_head = stationary_head
        super().__init__(env)

    def action(self, action):
        if self._stationary_base:
            action[..., -1] = 0
            action[..., -2] = 0
        if self._stationary_torso:
            action[..., -3] = 0
        if self._stationary_head:
            action[..., -4] = 0
            action[..., -5] = 0
        return action
