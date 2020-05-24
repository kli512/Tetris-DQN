import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gym_tetris.envs.Tetris as Tetris

import itertools

# https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    ROWS = 40
    COLS = 10

    def __init__(self, seed=None, next_pieces=5, simple=True):
        if simple:
            self._generate_state = self._generate_simple_state
        else:
            self._generate_state = self._generate_complex_state

        self.seed = seed
        self.game = Tetris.Board(rseed=seed)
        self.next_pieces = next_pieces

        self._pieces = {
            'I': 0,
            'O': 1,
            'T': 2,
            'S': 3,
            'Z': 4,
            'J': 5,
            'L': 6,
        }

        # Making and mapping the action space to ints
        self.action_space = spaces.Discrete(8)
        self._actions = {
            0: 'hold',
            1: 'hd',
            2: 'd',
            3: 'l',
            4: 'r',
            5: 'cw',
            6: 'ccw',
            7: ''
        }

        # Making the observation state
        if simple:
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.game.playable_height, self.game.width, 2))
        else:
            self.observation_space = spaces.Tuple((
                spaces.Tuple(tuple([spaces.MultiBinary(self.game.width)] * 10)),                            # Board state
                spaces.Tuple((
                    spaces.Tuple((spaces.Discrete(self.game.height), spaces.Discrete(self.game.width))),    # cur piece location
                    spaces.Discrete(4),                                                                     # cur piece rotation
                    spaces.Discrete(7)                                                                      # cur piece type
                )),
                spaces.Discrete(8),                                                                         # Hold piece (can be none)
                spaces.Tuple(tuple([spaces.Discrete(7)] * self.next_pieces))                                # Next pieces
            ))

        self.last_max_height = 0

        self.viewer = None
        self.view_board = None
        self.view_state = [[False for c in range(10)] for r in range(20)]

    def step(self, action):
        iscore = self.game.score
        self.game.act(self._actions[action])
        reward = self.game.score - iscore

        base_reward = 0.04

        delta_height = max(26 - r - self.last_max_height, 0)
        reward -= delta_height * base_reward * 250

        reward += base_reward * 1 # incentivize not dying
        if self._actions[action] == 'd':
            reward += base_reward * 1
        if self._actions[action] == 'hd':
            reward += base_reward * 25

        for r in range(self.game.height):
            if any(self.game._board[r]):
                break



        # if self._actions[action] == 'l':
        #     reward = 1
        # else:
        #     reward = 0

        # _board is a 2d matrix representing the game
        # held_piece is a string. Convert to one hot by _pieces
        # next_pieces is a deque of strings. Convert to one hot by _pieces
        return self._generate_state(), reward, self.game.dead, {'action': self._actions[action]}

    def reset(self):
        self.game = Tetris.Board(rseed=self.seed)
        return self._generate_state()

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(400, 800)

            self.view_board = [[None] * 10 for r in range(20)]
            for r in range(20):
                for c in range(10):
                    bot = 700 - 20 * r
                    top = bot - 20
                    left = 100 + 20 * c
                    right = left + 20
                    self.view_board[r][c] = rendering.FilledPolygon([(left, bot), (left, top), (right, top), (right, bot)])
                    self.viewer.add_geom(self.view_board[r][c])

        for r in range(6, 26):
            for c in range(10):
                if self.game._board[r][c] == 1 or (r, c) in self.game.cur_piece.occupied():
                    if not self.view_state[r - 6][c]:
                        self.view_state[r - 6][c] = True
                        self.view_board[r - 6][c].set_color(0, 1, 0)
                else:
                    if self.view_state[r - 6][c]:
                        self.view_state[r - 6][c] = False
                        self.view_board[r - 6][c].set_color(0, 0, 0)

        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _generate_simple_state(self):
        return [self.game.state()[self.game.height - self.game.playable_height : self.game.height],
                self.game._board[self.game.height - self.game.playable_height : self.game.height]]



    def _generate_complex_state(self):
        hold = 8
        try:
            hold = self._pieces[self.game.held_piece]
        except KeyError:
            pass

        return (
            self.game._board,
            (
                tuple(self.game.cur_piece.pos),
                self.game.cur_piece.rotation,
                self._pieces[self.game.cur_piece.piece_str]
            ),                                                          # currently gives ((row, col), rotation, and piece type). can do occupied()?
            hold,
            tuple(itertools.islice(map(lambda x: self._pieces[x], self.game.next_pieces), 0, 5))
        )
