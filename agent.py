import torch
from torch import nn
import numpy as np
import scipy as scy

import human_interface

class Play:
    def __init__(self):
        self.interface = human_interface.Interface(human_player=False)
        self.episodic_memory = []

    def agent_game_loop(self):
        state = 0
        while state == 0:
            state, frame = self.interface.update_board()[0]
            self.episodic_memory.append(torch.tensor(frame))
        return state


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential()
