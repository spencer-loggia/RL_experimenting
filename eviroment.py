import numpy as np
import sys
from tkinter import Tk
import random
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

    def __init__(self, max_trail=15, board_width=50, board_height=50, num_goals=15, num_player=1):
        self.max_trail = [max_trail for i in range(num_player)]
        self.line_count = 0
        self.width = board_width
        self.height = board_height
        self.timeout = .01
        self.board_state = np.zeros((self.height, self.width))
        self.board_state[:2, :] = 1
        self.board_state[-2:, :] = 1
        self.board_state[:, :2] = 1
        self.board_state[:, -2:] = 1
        starts = [self.set_goal(edge_margin=5) for i in range(num_player)]
        self.goals = [self.set_goal() for i in range(num_goals)]
        self.cur_pos = [[start[0], start[1]] for start in starts]
        for p in self.cur_pos:
            self.board_state[tuple(p)] = 2
        for goal in self.goals: self.board_state[goal] = 1.5
        self.trail = [[] for i in range(num_player)]
        self.cur_direction = [random.choice(['r', 'l', 'u', 'd']) for i in range(num_player)]
        self.num_alive = num_player

    def set_goal(self, edge_margin=2):
        goal = (np.random.rand(2) * [self.height - (2*edge_margin), self.width - (2*edge_margin)]).astype(int)
        goal = (goal[0] + edge_margin, goal[1] + edge_margin)
        check = self.board_state[goal]
        if check != 0:
            goal = self.set_goal()
        return goal

    def move(self, direction, pid=0):
        self.trail[pid].append(self.cur_pos[pid])
        if direction == 'r' and self.cur_direction[pid] != 'l':
            self.board_state[tuple(self.cur_pos[pid])] = 1
            self.cur_pos[pid] = [self.cur_pos[pid][0], self.cur_pos[pid][1] + 1]
            self.cur_direction[pid] = direction
        elif direction == 'l' and self.cur_direction[pid] != 'r':
            self.board_state[tuple(self.cur_pos[pid])] = 1
            self.cur_pos[pid] = [self.cur_pos[pid][0], self.cur_pos[pid][1] - 1]
            self.cur_direction[pid] = direction
        elif direction == 'u' and self.cur_direction[pid] != 'd':
            self.board_state[tuple(self.cur_pos[pid])] = 1
            self.cur_pos[pid] = [self.cur_pos[pid][0] - 1, self.cur_pos[pid][1]]
            self.cur_direction[pid] = direction
        elif direction == 'd' and self.cur_direction[pid] != 'u':
            self.board_state[tuple(self.cur_pos[pid])] = 1
            self.cur_pos[pid] = [self.cur_pos[pid][0] + 1, self.cur_pos[pid][1]]
            self.cur_direction[pid] = direction
        else:
            return self.step(pid=pid)

        for i in range(len(self.goals)):
            if tuple(self.cur_pos[pid]) == self.goals[i]:
                self.goals[i] = self.set_goal()
                self.board_state[tuple(self.goals[i])] = 1.5
                self.max_trail[pid] += 2
                return -1
        if len(self.trail[pid]) > self.max_trail[pid]:
            self.board_state[tuple(self.trail[pid].pop(0))] = 0
        if self.board_state[tuple(self.cur_pos[pid])] == 1:
            for t in self.trail[pid]:
                self.board_state[tuple(t)] = 0
            self.num_alive -= 1
            return 1
        else:
            self.board_state[tuple(self.cur_pos[pid])] = 2
            return 0

    def step(self, pid=0):
        return self.move(self.cur_direction[pid], pid=pid)

