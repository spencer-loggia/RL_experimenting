from torch import nn
import numpy as np
from torch import functional as F
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self  # .cuda(0)
        self.conv1 = nn.Conv2d(1, 20, 5)  # .cuda(0)
        self.relu = nn.ReLU()  # .cuda(0)
        self.conv2 = nn.Conv2d(100, 50, 3, stride=2)  # .cuda(0)
        self.norm = nn.BatchNorm2d(20)  # .cuda(0)
        self.pool = nn.MaxPool2d(5)  # .cuda(0)
        self.drop = nn.Dropout(.3)  # .cuda(0)

        conv_out_dim = 720

        # self.ln1 = nn.Linear(conv_out_dim, 2500)  # .cuda(0)
        self.tanh = nn.Tanh()  # .cuda(0)
        # self.ln2 = nn.Linear(2500, 1250)  # .cuda(0)

        self.ln3 = nn.Linear(1260, 600)  # .cuda(0)
        self.ln4 = nn.Linear(600, 200)  # .cuda(0)
        self.ln5 = nn.Linear(200, 60)  # .cuda(0)
        self.ln6 = nn.Linear(60, 10)  # .cuda(0)
        self.ln7 = nn.Linear(10, 3)  # .cuda(0)
        self.ln8 = nn.Linear(6, 3)  # .cuda(0)
        self.soft = nn.Softmax()  # .cuda(0)

    def forward(self, x, prev_move, e=.5, training=True):
        torch.cuda.set_device(0)
        # self  # .cuda(0)

        x = x.view(-1, 1, x.shape[0], x.shape[1]).float()  # .cuda(0)

        x = self.conv1(x)  # .cuda(0)
        x = self.norm(x)  # .cuda(0)
        x = self.pool(x)  # .cuda(0)
        x = self.tanh(x)  # .cuda(0)

        x = x.reshape(1, -1)  # .cuda(0)

        x = self.ln3(x)  # .cuda(0)
        x = self.tanh(x)  # .cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training)  # .cuda(0)

        x = self.ln4(x)  # .cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training)  # .cuda(0)
        x = self.relu(x)  # .cuda(0)


        x = self.ln5(x)  # .cuda(0)
        x = self.tanh(x)  # .cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training)  # .cuda(0)

        x = self.ln6(x)  # .cuda(0)
        x = nn.functional.dropout2d(x, p=e, training=training)  # .cuda(0)
        x = self.relu(x)  # .cuda(0)

        y = self.ln7(x)  # .cuda(0)
        y = self.tanh(y)  # .cuda(0)

        prev_move = prev_move.data.reshape(1, -1)  # .cuda(0)
        y = torch.cat((y, prev_move), 1)  # .cuda(0)

        y = self.ln8(y)  # .cuda(0)
        y = self.relu(y)  # .cuda(0)

        action_probs = self.soft(y)  # .cuda(0)
        return action_probs.clone()  # .cuda(0)


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        # self  # .cuda(0)
        self.cn1 = nn.Conv2d(1, 10, 4, stride=2, padding=2)  # .cuda(0)
        self.lrelu = nn.LeakyReLU()  # .cuda(0)
        self.cn2 = nn.Conv2d(10, 20, 5, stride=2, padding=2)  # .cuda(0)
        self.cn3 = nn.Conv2d(20, 40, 3, stride=1)
        self.cn4 = nn.Conv2d(40, 80, 6, stride=3, padding=3)
        self.cn5 = nn.Conv2d(80, 160, 4, stride=2, padding=2)# .cuda(0)
        self.cn6 = nn.Conv2d(160, 320, 2)
        self.norm = nn.BatchNorm2d(20)  # .cuda(0)
        self.mpl = nn.MaxPool2d(3)  # .cuda(0)
        self.mpl2 = nn.MaxPool2d(2)  # .cuda(0)
        self.drop = nn.Dropout(.3)  # .cuda(0)

        # self.ln1 = nn.Linear(conv_out_dim, 2500)  # .cuda(0)
        self.tanh = nn.Tanh()  # .cuda(0)
        # self.ln2 = nn.Linear(2500, 1250)  # .cuda(0)

        self.cnfc1 = nn.Conv2d(320, 160, 1)  # .cuda(0)
        self.cnfc2 = nn.Conv2d(160, 60, 1)  # .cuda(0)
        self.cnfc3 = nn.Conv2d(60, 20, 1)  # .cuda(0)
        self.cnfc4 = nn.Conv2d(20, 4, 1)

    def forward(self, x, e=.5, batch_size=1, training=True):
        #torch.cuda.set_device(0)
        # self  # .cuda(0)

        x = x.view(batch_size, 1, x.shape[-2], x.shape[-1]).float()  # .cuda(0)

        x = self.cn1(x)  # .cuda(0)
        #x = self.mpl2(x)  # .cuda(0)
        x = self.lrelu(x)  # .cuda(0)

        x = self.cn2(x)  # .cuda(0)
        #x = self.mpl2(x)  # .cuda(0)
        x = self.lrelu(x)  # .cuda(0)

        x = self.cn3(x)  # .cuda(0)
        #x = self.mpl(x)  # .cuda(0)
        x = self.lrelu(x)  # .cuda(0)

        x = self.cn4(x)
        x = self.lrelu(x)

        x = self.cn5(x)
        x = self.lrelu(x)

        while x.shape[2] !=1 and x.shape[3] !=1:
            x = self.cn6(x)
            x = self.lrelu(x)

        x = self.cnfc1(x)  # .cuda(0)
        x = self.lrelu(x)  # .cuda(0)

        x = self.cnfc2(x)  # .cuda(0)
        x = self.lrelu(x)  # .cuda(0)

        x = self.cnfc3(x)  # .cuda(0)
        x = self.lrelu(x)

        y = self.cnfc4(x)

        y = y.reshape(batch_size, 4)

        return y.clone()  # .cuda(0)


