from . import chess_utils
import chess
from chess import Piece, square
from pettingzoo import AECEnv
from gym import spaces
import numpy as np
import warnings
from pettingzoo.utils.agent_selector import agent_selector
from autograph.lib.envs.utils import wrappers


def env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):

    metadata = {'render.modes': ['human'], "name": "chess_v2"}

    def __init__(self):
        super().__init__()

        self.board = chess.Board()

        self.agents = ["player_{}".format(i) for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self.observation_spaces = {name: spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.bool),
            'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
        }) for name in self.agents}


        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}
        self.shape = (8, 8)
        self.agent_selection = 'player_0'
        self._last_fen = self.board.fen(shredder=True)
        self._last_obs = chess_utils.get_observation(self.board, 'player_0')
        self._last_action_space = None
        self.turn_num = 0

    def observe(self, agent='player_0'):
        agent = self.agent_selection
        observation = chess_utils.get_observation(self.board, agent)
        legal_moves = chess_utils.legal_moves(self.board) if agent == self.agent_selection else []

        action_mask = np.zeros(4672, int)
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}



    def save_state(self):
        return self.board.fen(shredder=True), self.agent_selection, self.turn_num, \
               self.infos, self.dones.copy()

    def load_state(self, state):
        self.board.set_fen(state[0])
        self.agent_selection = state[1]
        self.turn_num = state[2]
        self.infos = state[3]
        self.dones = state[4].copy()

        if self._agent_selector.next() == self.agent_selection:
            pass
        else:
            self._agent_selector.next()

        self.unset_game_result()


    def load_state_from_tensor(self, confusing_tensor):
        self.reset()
        empty_board = np.zeros([8, 8], dtype=np.int8)
        array_of_board = confusing_tensor.cpu().detach().numpy().reshape((20, 8, 8))

        # TODO add results of these castling indicators to board params
        castling_rights_q_0 = True if array_of_board[0,0,0] == 1 else False
        castling_rights_k_0 = True if array_of_board[0,0,0] == 1 else False

        castling_rights_q_1 = True if array_of_board[2,0,0] == 1 else False
        castling_rights_k_1 = True if array_of_board[3,0,0] == 1 else False

        #player_num = 'player_0' if array_of_board[4,0,0] == 0 else 'player_1'
        #TODO: allow player_1 to go first....errors when we allow this
        #      line 229, in step
        #     assert chosen_move in self.board.legal_moves

        player_num = 'player_0'
        if player_num is 'player_1':
            self.agent_selection = self._agent_selector.next()
            print('player1 plays first !!!!!!!!!!!!!!!!')
        else:
            print('player0 plays first !!!!!!!!!!!!!!!!')
        # skip layers 5 (50 move) and layer 6 (all 1's for some reason)

        # now we add pieces.

        # get indices

        # use square(file, rank) to get squares; then add pieces to board

        board_dict = dict()
        def get_square_num(i, j):
            j = (7-j)
            return (j * 8) + i


        PAWNS = np.argwhere(1 == array_of_board[7])
        squares = (get_square_num(idx[1], idx[0]) for idx in PAWNS)
        board_dict.update({square: Piece.from_symbol('P') for square in squares})

        KNIGHTS = np.argwhere(1 == array_of_board[8])
        squares = (get_square_num(idx[1], idx[0]) for idx in KNIGHTS)
        board_dict.update({square: Piece.from_symbol('N') for square in squares})

        BISHOPS = np.argwhere(1 == array_of_board[9])
        squares = (get_square_num(idx[1], idx[0]) for idx in BISHOPS)
        board_dict.update({square: Piece.from_symbol('B') for square in squares})

        ROOKS = np.argwhere(1 == array_of_board[10])
        squares = (get_square_num(idx[1], idx[0]) for idx in ROOKS)
        board_dict.update({square: Piece.from_symbol('R') for square in squares})

        QUEENS = np.argwhere(1 == array_of_board[11])
        squares = (get_square_num(idx[1], idx[0]) for idx in QUEENS)
        board_dict.update({square: Piece.from_symbol('Q') for square in squares})

        KING = np.argwhere(1 == array_of_board[12])
        squares = (get_square_num(idx[1], idx[0]) for idx in KING)
        board_dict.update({square: Piece.from_symbol('K') for square in squares})
        #####################################################################################3
        pawns = np.argwhere(1 == array_of_board[13])
        squares = (get_square_num(idx[1], idx[0]) for idx in pawns)
        board_dict.update({square: Piece.from_symbol('p') for square in squares})

        knights = np.argwhere(1 == array_of_board[14])
        squares = (get_square_num(idx[1], idx[0]) for idx in knights)
        board_dict.update({square: Piece.from_symbol('n') for square in squares})

        bishops = np.argwhere(1 == array_of_board[15])
        squares = (get_square_num(idx[1], idx[0]) for idx in bishops)
        board_dict.update({square: Piece.from_symbol('b') for square in squares})

        rooks = np.argwhere(1 == array_of_board[16])
        squares = (get_square_num(idx[1], idx[0]) for idx in rooks)
        board_dict.update({square: Piece.from_symbol('r') for square in squares})

        queens = np.argwhere(1 == array_of_board[17])
        squares = (get_square_num(idx[1], idx[0]) for idx in queens)
        board_dict.update({square: Piece.from_symbol('q') for square in squares})

        king = np.argwhere(1 == array_of_board[18])
        squares = (get_square_num(idx[1], idx[0]) for idx in king)
        board_dict.update({square: Piece.from_symbol('k') for square in squares})
        # memory of repetition in 19th (20th) channel
        repetition = array_of_board[19][1]

        '''for key, val in board_dict.items():
            self.board.set_piece_at(key, val)'''
        self.board.set_piece_map(board_dict)
        self._last_obs = chess_utils.get_observation(self.board, player_num)
        self._last_fen = self.board.fen()
        self.has_reset = True



    def reset(self):
        self.has_reset = True

        self.agents = self.possible_agents[:]

        self.board = chess.Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._last_fen = self.board.fen(shredder=True)
        self._last_obs = chess_utils.get_observation(self.board, self.agent_selection)
        next_legal_moves = chess_utils.legal_moves(self.board)
        self.turn_num = 0
        return self._last_obs, self._last_fen, next_legal_moves

    def get_fen(self):
        return self._last_fen

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.dones[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def unset_game_result(self):
        for i, name in enumerate(self.agents):
            self.dones[name] = False
            self.rewards[name] = 0

    def step(self, action):
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        if self.dones[self.agent_selection]:
            self._last_obs = chess_utils.get_observation(self.board, self.agent_selection, dontRemove=True)
            next_legal_moves = chess_utils.legal_moves(self.board)
            self._last_fen = self.board.fen(shredder=True)
            return (self._last_obs, self._last_fen, next_legal_moves), \
                   [self.rewards[self.agent_selection], self.rewards[self._agent_selector.next()]], \
                   True, {'player_moved': current_agent, 'move': None}


        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        #print('chosenCHOSEN MOVE: ', chosen_move)

        #print('LEGAL MOVES', self.board.legal_moves)

        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)
        self.agent_selection = self._agent_selector.next()
        next_legal_moves = chess_utils.legal_moves(self.board)

        # claim draw is set to be true to allign with normal tournament rules
        is_repetition = self.board.is_repetition(3)
        is_50_move_rule = self.board.can_claim_fifty_moves()
        is_claimable_draw = is_repetition or is_50_move_rule

        self._accumulate_rewards()
        self._last_fen = self.board.fen(shredder=True)
        #TODO : change back to self.agent_selection
        self._last_obs = chess_utils.get_observation(self.board, self.agent_selection, dontRemove=True)
        is_stale_or_checkmate = not any(next_legal_moves)
        game_over = is_claimable_draw or is_stale_or_checkmate
        rewards_ret = [0, 0]
        if game_over:
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)
            rewards_ret = [self.rewards[current_agent], self.rewards[self.agent_selection]]

        '''self._last_fen = self.board.fen(shredder=True)
        self._last_obs = chess_utils.get_observation(self.board, self.agent_selection, dontRemove=True)'''
        self.turn_num += 1
        #       last obs, rewards, game over, info


        return (self._last_obs, self._last_fen, next_legal_moves), \
               rewards_ret, game_over, {'player_moved': current_agent, 'move': chosen_move}

    def render(self, mode='human'):
        print(f"game_turn_num: {self.turn_num}")
        print(self.board)
        print("------------------")

        return str(self.board)




    def close(self):
        pass
