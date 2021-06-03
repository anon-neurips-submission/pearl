from .base import BaseWrapper
from ..env_logger import EnvLogger


class TerminateIllegalWrapper(BaseWrapper):
    '''
    this wrapper terminates the game with the current player losing
    in case of illegal values

    parameters:
        - illegal_reward: number that is the value of the player making an illegal move.
    '''
    def __init__(self, env, illegal_reward):
        super().__init__(env)
        self._illegal_value = illegal_reward
        self._prev_obs = None

    def save_state(self):
        return self._illegal_value, self._prev_obs, super().save_state()


    def load_state(self, state):
        self._illegal_value = state[0]
        self._prev_obs = state[1]
        return super().load_state(state[2])

    def reset(self):
        self._terminated = False
        self._prev_obs = None
        return super().reset()

    def observe(self, agent):
        obs = super().observe(agent)
        self._prev_obs = obs
        return obs

    def step(self, action):
        current_agent = self.agent_selection
        if self._prev_obs is None:
            self.observe(self.agent_selection)
        assert 'action_mask' in self._prev_obs, "action_mask must always be part of environment observation as an element in a dictionary observation to use the TerminateIllegalWrapper"
        _prev_action_mask = self._prev_obs['action_mask']
        _prev_obs = self._prev_obs
        self._prev_obs = None
        if self._terminated and self.dones[self.agent_selection]:
            if current_agent == 'player_1':
                other = 'player_0'
            else:
                other = 'player_1'

            reward_ret = [self.rewards[current_agent], self.rewards[other]]

            for k, v in self.rewards.items():
                if v != self.rewards[current_agent]:
                    reward_ret.append(v)

            return _prev_obs, reward_ret, self._terminated, dict()
        elif not self.dones[self.agent_selection] and not _prev_action_mask[action]:
            EnvLogger.warn_on_illegal_move()

            print("Illegal action was ", action, " by " , self.agent_selection)

            self._cumulative_rewards[self.agent_selection] = 0
            self.dones = {d: True for d in self.dones}
            self._prev_obs = None
            self.rewards = {d: 0 for d in self.dones}
            self.rewards[current_agent] = float(self._illegal_value)
            self._accumulate_rewards()
            self._dones_step_first()
            self._terminated = True

            reward_ret = [self.rewards[current_agent]]

            for k, v in self.rewards.items():
                if v != self.rewards[current_agent]:
                    reward_ret.append(v)

            return _prev_obs, reward_ret, self._terminated, dict()
        else:
            return super().step(action)

    def __str__(self):
        return str(self.env)
