import numpy as np
import sys
from tkinter import Tk

import matplotlib.pyplot as plt

class RunnerEnv:

    def __init__(self, diff=.6, board_width=50, board_height=100):
        """
        each square has a 1/30 * diff chance of becoming a barrier source
        once a barrier source is chosen it has a 1/2 chance of extending in both directions, until
        it is not extended.
        :param diff: The difficulty level
        """
        self.diff = diff
        self.line_count = 0
        self.width = board_width
        self.height = board_height
        self.timeout = .01
        self.board_state = []
        for i in range(int(self.height/4)):
            line = np.zeros(self.width)
            line[0] = 1
            line[self.width - 1] = 1
            self.board_state.append(line)
        for i in range(int(3*self.height/4)):
            self.board_state.append(self.generate_line())
        self.board_state[5][int(self.width/2)] = 2

    def generate_line(self) -> np.ndarray:
        dist = np.random.uniform(size=self.width)
        threshold = float(1/30)*self.diff
        sources = np.argwhere(dist < threshold)
        line = np.zeros(self.width)
        line[sources] = 1
        self.line_count += 1
        for i in range(self.width):
            if line[i] == 1:
                num = np.random.uniform()
                if num <= .5 and i+1 < self.width:
                    line[i+1] = 1
        line[0] = 1
        line[1] = 1
        line[self.width - 1] = 1
        line[self.width - 2] = 1
        return line

    def move(self, direction):
        if direction == 'l':
            return self.move_left()
        elif direction == 'r':
            return self.move_right()
        else:
            return 0

    def move_left(self):
        index = np.argwhere(self.board_state[5] == 2)
        if index - 1 >= 0:
            self.board_state[5][index] = 0
            self.board_state[5][index - 1] = 2
            return 0
        return 1

    def move_right(self):
        index = np.argwhere(self.board_state[5] == 2)
        if index + 1 < self.width:
            self.board_state[5][index] = 0
            self.board_state[5][index + 1] = 2
            return 0
        return 1

    def step(self):
        nline = self.generate_line()
        self.board_state.append(nline)
        index = np.argwhere(self.board_state[5] == 2)
        self.board_state[5][index] = 0
        if self.board_state[6][index] == 1:
            return -1
        self.board_state[6][index] = 2
        self.board_state.pop(0)
        return 0


class SnakeEnv:

    def __init__(self, max_trail=15, board_width=50, board_height=100):
        self.max_trail = max_trail
        self.line_count = 0
        self.width = board_width
        self.height = board_height
        self.timeout = .01
        self.board_state = np.zeros((self.height, self.width))
        self.board_state[:2, :] = 1
        self.board_state[-2:, :] = 1
        self.board_state[:, :2] = 1
        self.board_state[:, -2:] = 1
        start = (np.random.rand(2) * [self.height - 10, self.width - 10]).astype(int)
        self.goal = self.set_goal()
        self.cur_pos = (start[0] + 5, start[1] + 5)
        self.board_state[self.cur_pos] = 2
        self.board_state[self.goal] = 1.5
        self.trail = []
        self.cur_direction = 'r'

    def set_goal(self):
        goal = (np.random.rand(2) * [self.height - 4, self.width - 4]).astype(int)
        goal = (goal[0] + 2, goal[1] + 2)
        check = self.board_state[goal]
        if check != 0:
            goal = self.set_goal()
        return goal

    def move(self, direction):
        self.board_state[self.cur_pos] = 1
        self.trail.append(self.cur_pos)
        if direction == 'r':
            self.cur_pos = (self.cur_pos[0], self.cur_pos[1] + 1)
        elif direction == 'l':
            self.cur_pos = (self.cur_pos[0], self.cur_pos[1] - 1)
        elif direction == 'u':
            self.cur_pos = (self.cur_pos[0] - 1, self.cur_pos[1])
        elif direction == 'd':
            self.cur_pos = (self.cur_pos[0] + 1, self.cur_pos[1])
        else:
            print('Bad action given', sys.stderr)
            raise KeyError
        self.cur_direction = direction
        if self.cur_pos == self.goal:
            self.goal = self.set_goal()
            self.board_state[self.goal] = 1.5
            self.max_trail += 3
            return -1
        if len(self.trail) > self.max_trail:
            self.board_state[self.trail.pop(0)] = 0
        if self.board_state[self.cur_pos] == 1:
            return 1
        else:
            self.board_state[self.cur_pos] = 2
            return 0

    def step(self):
        return self.move(self.cur_direction)

