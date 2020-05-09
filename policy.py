from torch import nn
import numpy as np
from torch import functional as F
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(100, 50, 3, stride=2)
        self.norm = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(5)
        self.drop = nn.Dropout(.3)

        conv_out_dim = 720

        self.ln1 = nn.Linear(conv_out_dim, 2500)
        self.tanh = nn.Tanh()
        self.ln2 = nn.Linear(2500, 1250)

        self.ln3 = nn.Linear(1260, 600)
        self.ln4 = nn.Linear(600, 300)
        self.ln5 = nn.Linear(300, 100)
        self.ln6 = nn.Linear(100, 20)
        self.ln7 = nn.Linear(20, 3)
        self.soft = nn.Softmax()

    def forward(self, x, prev_move, e=.5, training=True):
        x = x.view(-1, 1, x.shape[0], x.shape[1]).float()

        x = self.conv1(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.tanh(x)

        x = x.reshape(1, -1)

        prev_move = prev_move.reshape(1, -1)

        x = self.ln3(x)
        x = self.tanh(x)
        x = nn.functional.dropout2d(x, p=e, training=training)

        x = self.ln4(x)
        x = self.relu(x)
        x = nn.functional.dropout2d(x, p=e, training=training)

        x = self.ln5(x)
        x = self.tanh(x)
        x = nn.functional.dropout2d(x, p=e, training=training)

        x = self.ln6(x)
        x = self.relu(x)
        x = nn.functional.dropout2d(x, p=e, training=training)

        x = self.ln7(x)
        x = x + prev_move
        action_probs = self.soft(x)
        return action_probs

