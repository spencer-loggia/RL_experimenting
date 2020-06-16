from torch import nn
import numpy as np
from torch import functional as F
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.cuda(0)
        self.conv1 = nn.Conv2d(1, 20, 5).cuda(0)
        self.relu = nn.ReLU().cuda(0)
        self.conv2 = nn.Conv2d(100, 50, 3, stride=2).cuda(0)
        self.norm = nn.BatchNorm2d(20).cuda(0)
        self.pool = nn.MaxPool2d(5).cuda(0)
        self.drop = nn.Dropout(.3).cuda(0)

        conv_out_dim = 720

        # self.ln1 = nn.Linear(conv_out_dim, 2500).cuda(0)
        self.tanh = nn.Tanh().cuda(0)
        # self.ln2 = nn.Linear(2500, 1250).cuda(0)

        self.ln3 = nn.Linear(1260, 600).cuda(0)
        self.ln4 = nn.Linear(600, 200).cuda(0)
        self.ln5 = nn.Linear(200, 60).cuda(0)
        self.ln6 = nn.Linear(60, 10).cuda(0)
        self.ln7 = nn.Linear(10, 3).cuda(0)
        self.ln8 = nn.Linear(6, 3).cuda(0)
        self.soft = nn.Softmax().cuda(0)

    def forward(self, x, prev_move, e=.5, training=True):
        torch.cuda.set_device(0)
        self.cuda(0)

        x = x.view(-1, 1, x.shape[0], x.shape[1]).float().cuda(0)

        x = self.conv1(x).cuda(0)
        x = self.norm(x).cuda(0)
        x = self.pool(x).cuda(0)
        x = self.tanh(x).cuda(0)

        x = x.reshape(1, -1).cuda(0)

        x = self.ln3(x).cuda(0)
        x = self.tanh(x).cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training).cuda(0)

        x = self.ln4(x).cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training).cuda(0)
        x = self.relu(x).cuda(0)


        x = self.ln5(x).cuda(0)
        x = self.tanh(x).cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training).cuda(0)

        x = self.ln6(x).cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training).cuda(0)
        x = self.relu(x).cuda(0)

        y = self.ln7(x).cuda(0)
        y = self.tanh(y).cuda(0)

        prev_move = prev_move.data.reshape(1, -1).cuda(0)
        y = torch.cat((y, prev_move), 1).cuda(0)

        y = self.ln8(y).cuda(0)
        y = self.relu(y).cuda(0)

        action_probs = self.soft(y).cuda(0)
        return action_probs.cuda(0).clone()


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.cuda(0)
        self.cn1 = nn.Conv2d(1, 40, 4, stride=2, padding=2).cuda(0)
        self.lrelu = nn.LeakyReLU().cuda(0)
        self.cn2 = nn.Conv2d(40, 80, 5, stride=1, padding=4).cuda(0)
        self.cn3 = nn.Conv2d(80, 150, 4, stride=2, padding=2).cuda(0)
        self.norm = nn.BatchNorm2d(20).cuda(0)
        self.mpl = nn.MaxPool2d(2).cuda(0)
        self.mpl2 = nn.MaxPool2d(2).cuda(0)
        self.drop = nn.Dropout(.3).cuda(0)

        # self.ln1 = nn.Linear(conv_out_dim, 2500).cuda(0)
        self.tanh = nn.Tanh().cuda(0)
        # self.ln2 = nn.Linear(2500, 1250).cuda(0)

        self.fc1 = nn.Linear(600, 200).cuda(0)
        self.fc2 = nn.Linear(200, 50).cuda(0)
        self.fc3 = nn.Linear(50, 3).cuda(0)

    def forward(self, x, e=.5, training=True):
        torch.cuda.set_device(0)
        self.cuda(0)

        x = x.view(-1, 1, x.shape[0], x.shape[1]).float().cuda(0)

        x = self.cn1(x).cuda(0)
        x = self.mpl2(x).cuda(0)
        x = self.lrelu(x).cuda(0)

        x = self.cn2(x).cuda(0)
        x = self.mpl2(x).cuda(0)
        x = self.lrelu(x).cuda(0)

        x = self.cn3(x).cuda(0)
        x = self.mpl2(x).cuda(0)
        x = self.lrelu(x).cuda(0)

        x = x.reshape(1, -1).cuda(0)

        x = self.fc1(x).cuda(0)
        x = self.lrelu(x).cuda(0)
        x = nn.functional.dropout(x, p=e, training=training).cuda(0)

        x = self.fc2(x).cuda(0)
        y = self.lrelu(x).cuda(0)
        y = next_hidden = nn.functional.dropout(y, p=e, training=training).cuda(0)

        y = self.fc3(y).cuda(0)
        y = self.lrelu(y).cuda(0)

        reward = y
        return reward.cuda(0).clone()


