from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gym import spaces
from . import go
from . import coords
import numpy as np
from autograph.lib.envs.utils.wrappers import capture_stdout, terminate_illegal, \
    assert_out_of_bounds, order_enforcing


def env(**kwargs):
    env = raw_env(**kwargs)
    env = capture_stdout.CaptureStdoutWrapper(env)
    env = terminate_illegal.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = assert_out_of_bounds.AssertOutOfBoundsWrapper(env)
    env = order_enforcing.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):

    metadata = {'render.modes': ['human'], "name": "go_v2"}

    def __init__(self, board_size: int = 7, komi: float = 2, board=None):
        # board_size: a int, representing the board size (board has a board_size x board_size shape)
        # komi: a float, representing points given to the second player.
        super().__init__()

        self._overwrite_go_global_variables(board_size=board_size)
        self._komi = komi

        self.agents = ['black_0', 'white_0']
        self.possible_agents = self.agents[:]
        self.has_reset = False

        self.observation_spaces = self._convert_to_dict(
            [spaces.Dict({'observation': spaces.Box(low=0, high=1, shape=(self._N, self._N, 3), dtype=np.bool),
                          'action_mask': spaces.Box(low=0, high=1, shape=((self._N * self._N) + 1,), dtype=np.int8)})
             for _ in range(self.num_agents)])

        self.action_spaces = self._convert_to_dict([spaces.Discrete(self._N * self._N + 1) for _ in range(self.num_agents)])
        self.action_space = self.action_spaces['black_0']
        self._agent_selector = agent_selector(self.agents)
        self.shape = (board_size, board_size)
        if board:
            # at some point we ALSO must convert the binary mask into the -1 0 1 board it wants

            self.reset_with_board(board)

    def _overwrite_go_global_variables(self, board_size: int):
        self._N = board_size
        go.N = self._N
        go.ALL_COORDS = [(i, j) for i in range(self._N) for j in range(self._N)]
        go.EMPTY_BOARD = np.zeros([self._N, self._N], dtype=np.int8)
        go.NEIGHBORS = {(x, y): list(filter(self._check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])) for x, y in go.ALL_COORDS}
        go.DIAGONALS = {(x, y): list(filter(self._check_bounds, [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)])) for x, y in go.ALL_COORDS}
        return

    def _check_bounds(self, c):
        return 0 <= c[0] < self._N and 0 <= c[1] < self._N

    def _encode_player_plane(self, agent):
        if agent == self.possible_agents[0]:
            return np.zeros([self._N, self._N], dtype=np.bool)
        else:
            return np.ones([self._N, self._N], dtype=np.bool)

    def _encode_board_planes(self, agent):
        agent_factor = -1 if agent == self.possible_agents[0] else 1
        current_agent_plane_idx = np.where(self._go.board == agent_factor)
        opponent_agent_plane_idx = np.where(self._go.board == -agent_factor)
        current_agent_plane = np.zeros([self._N, self._N], dtype=np.bool)
        opponent_agent_plane = np.zeros([self._N, self._N], dtype=np.bool)
        current_agent_plane[current_agent_plane_idx] = 1
        opponent_agent_plane[opponent_agent_plane_idx] = 1
        return current_agent_plane, opponent_agent_plane

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _encode_legal_actions(self, actions):
        return np.where(actions == 1)[0]

    def _encode_rewards(self, result):
        return [1, -1] if result == 1 else [-1, 1]

    def observe(self, agent=None):

        if not agent:
            agent = self.agent_selection

        current_agent_plane, opponent_agent_plane = self._encode_board_planes(agent)
        player_plane = self._encode_player_plane(agent)
        observation = np.stack((current_agent_plane, opponent_agent_plane, player_plane))

        legal_moves = self.next_legal_moves if agent == self.agent_selection else []
        #print(legal_moves)
        action_mask = np.zeros((self._N * self._N) + 1, int)
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        self._go = self._go.play_move(coords.from_flat(action))
        self._last_obs = self.observe(self.agent_selection)
        next_player = self._agent_selector.next()
        this_player = self._agent_selector.next()
        next_player = self._agent_selector.next()

        if self._go.is_game_over():
            self.dones = self._convert_to_dict([True for _ in range(self.num_agents)])
            self.rewards = self._convert_to_dict(self._encode_rewards(self._go.result()))
            self.next_legal_moves = [self._N * self._N]
            print('game over results:' , self._go.result_string())
        else:
            self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())

        self.agent_selection = next_player if next_player else self._agent_selector.next()
        self._accumulate_rewards()

        reward = 0 if not self._go.is_game_over() else self.rewards[this_player]

        rewards_ret = [reward, self.rewards[next_player]]

        if reward != 0:
            x = 5

        self._last_obs['action_mask'] = np.zeros(self._N * self._N + 1, dtype=int)

        self._last_obs['action_mask'][self.next_legal_moves] = 1
        # returning next_state, reward from this agent, if the game is over, and info dictionary (empty rn)
        return self._last_obs, rewards_ret, self._go.is_game_over(), dict()

    # where board is an np.array. 0's for empty. 1's for black. -1's for white.
    def reset_with_board(self, board, turn=0):
        self.has_reset = True
        self._go = go.Position(board=board, komi=self._komi)
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if turn == 1:
            self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{} for _ in range(self.num_agents)])
        self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())
        self._last_obs = self.observe(self.agents[0])
        return self._last_obs


    def load_state_from_tensor(self, confusing_tensor):
        empty_board = np.zeros([self._N, self._N], dtype=np.int8)
        array_of_board = confusing_tensor.cpu().detach().numpy().reshape((3, self._N, self._N))

        my_stones = array_of_board[0]
        your_stones = array_of_board[1]
        turn = array_of_board[2]

        indexes_of_my_stones = my_stones.nonzero()
        indexes_your_stones = your_stones.nonzero()

        empty_board[indexes_of_my_stones] = 1
        empty_board[indexes_your_stones] = -1

        turn = turn[1, 1]

        return self.reset_with_board(empty_board, turn)

    def save_state(self):
        # return go board, last obs, turn
        return self._go, \
               self.agent_selection, \
                self._cumulative_rewards, \
                self.rewards, \
                self.dones, \
                self.infos, \
                self.next_legal_moves, \
                self._last_obs

    def load_state(self, state):
        self._go = state[0]
        self.agent_selection = state[1]
        self._cumulative_rewards=state[2]
        self.rewards=state[3]
        self.dones=state[4]
        self.infos=state[5]
        self.next_legal_moves=state[6]
        self._last_obs=state[7]
        return self._last_obs

    def reset(self):
        self.has_reset = True
        self._go = go.Position(board=None, komi=self._komi)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{} for _ in range(self.num_agents)])
        self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())
        self._last_obs = self.observe(self.agents[0])
        return self._last_obs

    def render(self, mode='human'):
        print(self._go)

    def close(self):
        pass
