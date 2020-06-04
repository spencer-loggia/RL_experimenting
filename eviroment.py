import numpy as np
from tkinter import Tk

import matplotlib.pyplot as plt

class Env:

    def __init__(self, diff=.3, board_width=50, board_height=100):
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

    def step(self, command):
        nline = self.generate_line()
        self.board_state.append(nline)
        index = np.argwhere(self.board_state[5] == 2)
        self.board_state[5][index] = 0
        if self.board_state[6][index] == 1:
            return -1
        self.board_state[6][index] = 2
        self.board_state.pop(0)
        return 0

