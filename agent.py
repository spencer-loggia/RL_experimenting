import torch
from torch import nn
import numpy as np
import pickle as pk
from policy import Network
import human_interface


class Agent:
    def __init__(self):
        self.interface = None
        self.model = Network().float().cuda(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001)
        self.cur_frame = None

    def agent_game_loop(self):
        state = 0
        while state == 0:
            state, frame = self.interface.update_board()[0]

    def train(self, NUM_GAMES):
        data_file = open('./data/scores.csv', 'w')
        data_file.write("iter,score\n")
        self.model = self.model.float()
        for i in range(NUM_GAMES): 
            if (i+1) % 500 == 0:
                file = open('./models/trained' + str(i+1) + '.pkl', 'wb')
                torch.save(self.model, file)
                file.close()
            if i % 50 == 0:
                self.interface = human_interface.Interface(human_player=False)
            else:
                self.interface = human_interface.Interface(human_disp=False, human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)[5:45, :]
            self.optimizer.zero_grad()
            score = self.recursive_train(torch.zeros(3), i / NUM_GAMES, 0)
            data_file.write(str(i) + ',' + str(score) + '\n')
            print("score on game " + str(i) + ": " + str(score))


    # initialize recursion
    def recursive_train(self, prev, completion_ratio, count):
        count += 1
        game_over = False
        x = torch.from_numpy(self.cur_frame).cuda(0)
        logits = self.model(x.float(), prev.float(), e=max(.4, completion_ratio), training=True)
        cpu_logits = logits.cpu()
        if count % 50 == 0:
            print("move " + str(count) + " probs:  " + str(logits))
        epsilon = np.random.random(1)
        if epsilon <= max(.7, completion_ratio):
            move = torch.argmax(cpu_logits)
        else:
            # explore
            move = np.random.choice([0, 1, 2], p=cpu_logits.data.numpy().reshape(3))
        if move == 1:
            if 1 == self.interface.E.move_left():
                game_over = True
        elif move == 2:
            if 1 == self.interface.E.move_right():
                game_over = True
        # if move is 1 go straight

        state, self.cur_frame = self.interface.update_board()
        if self.cur_frame is not None:
            self.cur_frame = self.cur_frame[5:45, :]

        # determine results of this move
        # Set Ys to penalize death, reward staying alive
        # Also weight loss more when dying, since this is a rare important event
        if state != 0 or game_over:
            coef = 1
            game_over = True
            y = torch.ones(3).cuda(0)
            y[move] = 0
            y = y.reshape(1, -1)
            loss = coef * nn.functional.binary_cross_entropy(logits, y)
        else:  # state == 0
            y = torch.zeros(3).cuda(0)
            y[move] = 1
            y = y.reshape(1, -1)
            loss = 0.0001*nn.functional.binary_cross_entropy(logits, y)

        # DO NOT zero grad. Allow time propagation
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if game_over:
            return state
        else:
            return self.recursive_train(logits, completion_ratio, count)


torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
bob = Agent()
# file = open('./models/trained.pkl', 'rb')
# bob.model = pk.load(file).cuda(0)
bob.train(NUM_GAMES=1000)

torch.cuda.device_count()



