import torch
from torch import nn
import numpy as np
import scipy as scy
import pickle as pk
from policy import Network
import human_interface


class Agent:
    def __init__(self):
        self.interface = None
        self.model = Network().float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.005)
        self.cur_frame = None

    def agent_game_loop(self):
        state = 0
        while state == 0:
            state, frame = self.interface.update_board()[0]

    def train(self, NUM_GAMES):
        self.model = self.model.float()
        for i in range(NUM_GAMES):
            if i % 100 == 0:
                self.interface = human_interface.Interface(human_player=False)
            else:
                self.interface = human_interface.Interface(human_disp=False, human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)[5:45, :]
            self.optimizer.zero_grad()
            print("score on game " + str(i) + ": " + str(self.recursive_train(torch.zeros(3), i / NUM_GAMES)))
        file = open('./models/trained.pkl', 'wb')
        pk.dump(self.model, file)

    # initialize recursion
    def recursive_train(self, prev, completion_ratio):
        game_over = False
        x = torch.from_numpy(self.cur_frame)
        self.model = self.model.float()
        logits = self.model(x.float(), prev.float(), e=max(.4, completion_ratio), training=True)
        move = np.random.choice([0, 1, 2], p=logits.data.numpy().reshape(3))
        if move == 0:
            self.interface.E.move_left()
        elif move == 2:
            self.interface.E.move_right()
        # if move is 1 go straight

        state, self.cur_frame = self.interface.update_board()
        if self.cur_frame is not None:
            try:
                self.cur_frame = self.cur_frame[5:45, :]
            except IndexError:
                print('temp')

        # determine results of this move
        # Set Ys to penalize death, reward staying alive
        # Also weight loss more when dying, since this is a rare important event
        if state != 0:
            game_over = True
            y = torch.ones(3)
            y[move] = 0
            y = y.reshape(1, -1)
            loss = nn.functional.binary_cross_entropy(logits, y)
        # else:  # state == 0
        #     y = torch.zeros(3)
        #     y[move] = 1
        #     y = y.reshape(1, -1)
        #     loss = 0.01*nn.functional.binary_cross_entropy(logits, y)

            # DO NOT zero grad. Allow time propagation
            loss.backward(retain_graph=True)
            self.optimizer.step()

        if game_over:
            return state
        else:
            return self.recursive_train(logits, completion_ratio)


bob = Agent()
bob.train(NUM_GAMES=50000)





