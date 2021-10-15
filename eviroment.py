import numpy as np
import sys
from tkinter import Tk
import random
import matplotlib.pyplot as plt
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

    def __init__(self, input_layout: str, num_goals: int = 10, hp=100, num_players: int = 1):
        from PIL import Image
        layout_file = Image.open(input_layout)
        self.board_state = zoom(np.array(layout_file).mean(axis=2), .25)
        self.num_players = num_players
        med_val = np.mean(self.board_state.flatten())
        self.board_state[self.board_state <= med_val / .95] = .5
        self.board_state[self.board_state > med_val] = 0
        self.height, self.width = self.board_state.shape
        self.hp = hp
        self.board_state[0][self.board_state[0] == 0] = .5
        self.home = np.nonzero(self.board_state[1] == 0)[0]  # always at top
        self.goals = {self.set_goal() for _ in range(num_goals)}
        self.cur_pos = [(1, np.random.choice(self.home)) for _ in range(num_players)] # cur_pos denotes the top left corner of the 4 pixel agent
        for loc in self.cur_pos:
            self.board_state[loc[0]:min(loc[0]+2, self.height), loc[1]:min(loc[1] + 2, self.width)] = 1
        self.cur_direction = ['d' for _ in range(num_players)]
        self.num_alive = num_players
        r_coord = np.tile(np.arange(self.height), (self.width, 1)).T
        c_coord = np.tile(np.arange(self.width), (self.height, 1))
        self._loc_arr = np.stack([r_coord, c_coord], axis=2)

    def set_goal(self, edge_margin=2):
        goal = (np.random.rand(2) * [self.height - (2*edge_margin), self.width - (2*edge_margin)]).astype(int)
        goal = (goal[0] + edge_margin, goal[1] + edge_margin)
        check = self.board_state[goal]
        if check != 0:
            goal = self.set_goal()
        self.board_state[goal] = .8
        return goal

    def _preform_check_move(self, request, pid, direction):
        request = tuple(request)
        loc = self.cur_pos[pid]
        self.board_state[loc[0]:min(loc[0] + 2, self.height), loc[1]:min(loc[1] + 2, self.width)] = 0
        if .5 not in (self.board_state[request[0]:min(request[0] + 2, self.height), request[1]:min(request[1] + 2, self.width)]):
            self.cur_pos[pid] = request
            self.cur_direction[pid] = direction
        loc = self.cur_pos[pid]
        self.board_state[loc[0]:min(loc[0] + 2, self.height), loc[1]:min(loc[1] + 2, self.width)] = 1

    def observable_env(self, pid=0, vis_res=32):
        """
        Depends on sensory mode. default can see luminance of vector of 10 pixels across in front
        :param pid:
        :param vis_res: the visual resolution. Must be a power of 2
        :return:
        """
        direction = self.cur_direction[pid]
        if direction == 'r':
            rotation_size = -90
            eye_pos = np.array([[1, -1], [1, 0]])
        elif direction == 'l':
            rotation_size = 90
            eye_pos = np.array([[0, 0], [0, 1]])
        elif direction == 'u':
            rotation_size = 180
            eye_pos = np.array([[0, -1], [0, 0]])
        elif direction == 'd':
            rotation_size = 0
            eye_pos = np.array([[1, 0], [1, 1]])
        else:
            raise ValueError
        self.board_state[self.cur_pos[pid]] = .9
        rotated = np.round(rotate(self.board_state, rotation_size), decimals=1)
        r_cur_pos = np.nonzero(rotated == .9)
        r_loc_arr = self._loc_arr
        l_center = (int(r_cur_pos[0][0]) + eye_pos[0, 0], int(r_cur_pos[1][0]) + eye_pos[0, 1])
        r_center = (int(r_cur_pos[0][0]) + eye_pos[1, 0], int(r_cur_pos[1][0]) + eye_pos[1, 1])
        center_arr = np.array([list(l_center), list(r_center)])
        to_consider = r_loc_arr[max(l_center[0] + 1, r_center[0] + 1):, :, :]
        is_solid = rotated[max(l_center[0] + 1, r_center[0] + 1):, :] > 0.1
        solid_coords = to_consider[is_solid].astype(float)
        dists = cdist(solid_coords, center_arr + np.array([[1, 0], [1,  0]]))
        dists[dists == 0] = .001
        components = np.tile(solid_coords, (2, 1, 1)) - center_arr[:, None, :]
        components[components == 0] = .001
        val = components[:, :, 1] / components[:, :, 0]
        angles = np.round((np.degrees(np.arctan(val)) - 90) * -32)
        sort_idx = np.argsort(dists, axis=0)
        angles[0] = angles[0, sort_idx[:, 0]]
        angles[1] = angles[1, sort_idx[:, 1]]
        dists = dists.T
        dists[0] = dists[0, sort_idx[:, 0]]
        dists[1] = dists[1, sort_idx[:, 1]]
        sensor = np.zeros((2, 5760))
        visual_field = np.degrees(np.arctan(.5 / dists)) * 32
        max_dist = max(dists.flatten())
        for j in range(2):
            for i in range(len(angles[j])):
                l_bound = max(angles[j, i] - visual_field[j, i], 0)
                r_bound = min(angles[j, i] + visual_field[j, i], sensor.shape[1])
                lum_mod = min(r_bound - l_bound, 1) * (1 - (dists[j, i] / max_dist))
                l_bound = int(np.ceil(l_bound))
                r_bound = int(np.ceil(r_bound))
                mask = np.zeros(r_bound - l_bound)
                mask[sensor[j, l_bound:r_bound] == 0] = 1
                obs_idx = solid_coords[sort_idx[i, j]].astype(int)
                luminance = rotated[obs_idx[0], obs_idx[1]]
                sensor[j, l_bound:r_bound] += luminance * mask * lum_mod
        sensor = zoom(sensor, (1, 1 / 45), order=0)
        return sensor

    def move(self, direction, pid: int = 0):
        self.hp -= 1
        print(self.hp)
        if direction == 'r':
            request = [self.cur_pos[pid][0], self.cur_pos[pid][1] + 1]
            self._preform_check_move(request, pid, direction)
        elif direction == 'l':
            request = [self.cur_pos[pid][0], self.cur_pos[pid][1] - 1]
            self._preform_check_move(request, pid, direction)
        elif direction == 'u':
            request = [self.cur_pos[pid][0] - 1, self.cur_pos[pid][1]]
            self._preform_check_move(request, pid, direction)
        elif direction == 'd':
            request = [self.cur_pos[pid][0] + 1, self.cur_pos[pid][1]]
            self._preform_check_move(request, pid, direction)
        else:
            return 0

        loc = tuple(self.cur_pos[pid])
        agent_coords = self._loc_arr[loc[0]:min(loc[0] + 2, self.height), loc[1]:min(loc[1] + 2, self.width)]
        agent_coords = agent_coords.reshape(-1, 2)
        agent_coords = {tuple(item) for item in agent_coords}
        intersect = agent_coords & self.goals
        if intersect:
            for item in intersect:
                self.goals.remove(item)
                self.goals.add(self.set_goal())
            self.hp += 20
            return -1
        if self.hp == 0:
            self.num_alive -= 1
            return 100
        return 0

    def step(self, pid=0):
        self.hp -= .25
        return 0
