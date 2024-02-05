import numpy as np
import sys
from tkinter import Tk
import random
import matplotlib.pyplot as plt
import scipy.ndimage
import torch
from scipy.ndimage import zoom, rotate
from scipy.spatial.distance import cdist


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


class GridWorldRevolution:

    def __init__(self, input_layout: str, max_goals: int = 20, hp=30, num_players: int = 1, board_dim=20,
                 abundance=.1):
        from PIL import Image
        layout_file = np.array(Image.open(input_layout))
        print("Input layout has size", layout_file.shape)
        if layout_file.shape[0] != layout_file.shape[1]:
            raise ValueError("Board is expected to be square.")
        factor = board_dim / layout_file.shape[0]
        self.board_state = zoom(layout_file.mean(axis=2), factor)
        print("Board has size", self.board_state.shape)
        self.num_players = num_players
        med_val = np.mean(self.board_state)
        # set the background color
        self.board_state[self.board_state <= med_val / .95] = .5
        self.board_state[self.board_state > med_val] = 0
        self.height, self.width = self.board_state.shape
        self.hp = hp
        self.board_state[0][self.board_state[0] == 0] = .5

        # reward state params
        self.abundance = abundance
        self.reward_freq_params = self._get_reward_freq_dist()
        self.max_goals = max_goals
        self.goals = {self.set_goal() for _ in range(15)}

        # set start positions
        self.cur_pos = []
        for _ in range(num_players):
            home = np.argwhere(self.board_state == 0)
            loc = home[np.random.randint(len(home))] # cur_pos denotes the top left corner of the 4 pixel agent
            loc = (min(board_dim - 4, loc[0]), min(board_dim - 4, loc[1]))
            loc = (max(4, loc[0]), max(4, loc[1]))
            self.cur_pos.append(loc)
            self.board_state[loc[0], loc[1]] = 1
        self.cur_direction = ['d' for _ in range(num_players)]
        self.num_alive = num_players
        r_coord = np.tile(np.arange(self.height), (self.width, 1)).T
        c_coord = np.tile(np.arange(self.width), (self.height, 1))
        self._loc_arr = np.stack([r_coord, c_coord], axis=2)

    def _get_reward_freq_dist(self):
        base = np.zeros_like(self.board_state)
        base[5, 5] = 1
        base[15, 15] = 1
        base = scipy.ndimage.gaussian_filter(base, sigma=3)
        base[:3, :3] = 0
        base[-3:, -3:] = 0
        base /= np.sum(base)
        return base

    def set_goal(self):
        # places a goal (rewarded) object onto the board.
        goal = np.random.choice(self.reward_freq_params.size, p=self.reward_freq_params.flatten(), size=(1,))
        goal_x = int(goal // self.board_state.shape[0])
        goal_y = int(goal % self.board_state.shape[0])
        goal = (goal_x, goal_y)
        #goal = (np.random.rand(2) * [self.height - (2*edge_margin), self.width - (2*edge_margin)]).astype(int)
        #goal = (goal[0] + edge_margin, goal[1] + edge_margin)
        check = self.board_state[goal]
        if check != 0:
            goal = self.set_goal()
        self.board_state[goal] = .8
        return goal

    def _preform_check_move(self, request, pid, direction):
        request = tuple(request)
        if 0 < request[0] + 1 < self.height and 0 < request[1] + 1 < self.width and .5 not in np.array(
                (self.board_state[request[0]:request[0] + 2,
                 request[1]:request[1] + 2])
        ).flatten():
            loc = self.cur_pos[pid]
            self.board_state[loc[0], loc[1]] = 0
            self.cur_pos[pid] = request
            self.cur_direction[pid] = direction
            loc = self.cur_pos[pid]
            self.board_state[loc[0], loc[1]] = 1
            return 0
        return 1

    def move(self, direction, pid: int = 0):
        if direction == 'r':
            request = [self.cur_pos[pid][0], self.cur_pos[pid][1] + 1]
            self.hp -= 1
            hit = self._preform_check_move(request, pid, direction)
        elif direction == 'l':
            request = [self.cur_pos[pid][0], self.cur_pos[pid][1] - 1]
            self.hp -= 1
            hit = self._preform_check_move(request, pid, direction)
        elif direction == 'u':
            request = [self.cur_pos[pid][0] - 1, self.cur_pos[pid][1]]
            self.hp -= 1
            hit = self._preform_check_move(request, pid, direction)
        elif direction == 'd':
            request = [self.cur_pos[pid][0] + 1, self.cur_pos[pid][1]]
            self.hp -= 1
            hit = self._preform_check_move(request, pid, direction)
        else:
            request = [self.cur_pos[pid][0], self.cur_pos[pid][1]]
            self.hp -= .5
            hit = self._preform_check_move(request, pid, direction)
        if hit:
            self.hp -= 1
        loc = tuple(self.cur_pos[pid])
        agent_coords = self._loc_arr[loc[0], loc[1]]
        agent_coords = agent_coords.reshape(-1, 2)
        agent_coords = {tuple(item) for item in agent_coords}
        intersect = agent_coords & self.goals

        # some chance of adding new goal state
        p = self.abundance * (1 - (len(self.goals) / self.max_goals)) / self.num_alive
        if np.random.rand() < p:
            self.goals.add(self.set_goal())

        # compute loss / penalty
        if intersect:
            for item in intersect:
                self.goals.remove(item)
            self.hp += 10
            return 1
        if self.hp <= 0:
            self.num_alive -= 1
            return -2
        return 0

    def step(self, pid=0):
        self.hp -= 1
        if self.hp <= 0:
            self.num_alive -= 1
            return -1
        return 0
