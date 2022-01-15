from typing import Union

import numpy as np
import torch
from torch import nn


class TDAMN(nn.Module):
    """
    Temporal Dynamic Associative Memory Network

    """

    @staticmethod
    def _bandpass(arr: torch.Tensor, high: float, low: float) -> torch.Tensor:
        arr[arr > high] = high
        arr[arr < low] = low
        arr = arr.clone()
        arr.requires_grad = True
        return arr

    def __init__(self, size: int, gamma_decay: float = .9, bias: float = 0, verbose=False):
        super().__init__()
        self.size = size
        # if resistance is low quickly return to baseline (poor local memory)
        # if the resist
        self.resistance = self._bandpass(torch.normal(.8, .01, size=[size], requires_grad=False),
                                         high=.99, low=.01)
        self.gamma = gamma_decay
        self.weight = nn.Parameter(self._bandpass(torch.normal(0, .25, size=[size, size]),
                                                  high=1, low=-1))
        self.baseline_voltage = torch.ones(size=[size]) * bias
        self.voltage = self.baseline_voltage.clone()
        self.verbose = verbose

        self.tanh = nn.Tanh()

    def input(self, x):
        """
        :param x: input pattern
        :return: update current voltage states given new input pattern
        """
        self.voltage = self.voltage * (1 - self.resistance) + x * self.resistance
        if self.verbose:
            print('voltage update, ', self.voltage)

    def reset_voltage_states(self):
        self.voltage = self.baseline_voltage.clone().detach()
        self.voltage.requires_grad = False

    def encode(self, gamma):
        """
        encode current voltage pattern
        :return:
        """
        if self.verbose:
            print('Current saliency (gamma):', gamma)
        with torch.no_grad():
            tan_v = self.tanh(self.voltage)
            similarity_matrix = torch.outer(tan_v, tan_v)
            self.weight = nn.Parameter(torch.log(gamma * torch.exp(self.weight) +
                                                 (1 - gamma) * torch.exp(similarity_matrix)))
            if self.verbose:
                print("weight update, ", self.weight)

    def poll(self):
        mem = self.tanh(self.weight @ self.voltage.T)
        if self.verbose:
            print("memory recovered, ", mem)
        return mem

    def default_mode_iteration(self, x: Union[None, np.ndarray], gamma_override=None):
        """
        default operation of cognition layer. Requires no gradient based optimization
        :param gamma_override:
        :param x: current available stimuli input. can be noise if no stimuli is available
        :return:
        """
        if gamma_override:
            gamma = gamma_override
        else:
            gamma = self.gamma
        self.input(x)
        self.encode(gamma=gamma)
        mem = self.poll()
        # self.input(mem)
        # self.encode(gamma=self.gamma)
        return mem

    def forward(self, xi, gamma_override=None):
        """
        Gradient based optimization program.
        :param gamma_override:
        :param xi: the input state
        :return:
        """
        h = self.default_mode_iteration(xi, gamma_override=gamma_override)
        return torch.cat([xi, h])
