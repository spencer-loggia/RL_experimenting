import torch
from torch import nn
import numpy as np
import pickle as pk
from policy import Network
from policy import ConvNetwork
import human_interface
import time
import sys

class Agent:
    def __init__(self, lr=.0001):
        self.interface = None
        self.model = ConvNetwork().float().cuda(0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.cur_frame = None

    def agent_game_loop(self):
        state = 0
        while state == 0:
            state, frame = self.interface.update_board()[0]

    def train(self, NUM_GAMES, max_drop=.5, max_explore=.4, mode='det', disp_iter=25, save_iter=500):
        if max_drop > 1 or max_explore > 1 or mode not in ['det', 'stoch', 'threshold']:
            print('Invalid Hyperparameter. Learning Aborted', file=sys.stderr)
            return
        data_file = open('./data/scores.csv', 'w')
        data_file.write("iter,score\n")
        self.model = self.model.float()
        for i in range(NUM_GAMES): 
            if (i+1) % save_iter == 0:
                file = open('./models/trained' + str(i+1) + '.pkl', 'wb')
                torch.save(self.model, file)
                file.close()
            if i % disp_iter == 0:
                self.interface = human_interface.Interface(human_player=False)
            else:
                self.interface = human_interface.Interface(human_disp=False, human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)[4:46, :]
            self.optimizer.zero_grad()
            score = self.recursive_train(torch.zeros(50), i / NUM_GAMES, 0, max_drop, max_explore, mode)
            data_file.write(str(i) + ',' + str(score) + '\n')
            print("score on game " + str(i) + ": " + str(score))

    # initialize recursion
    def recursive_train(self, prev, completion_ratio, count, max_drop, max_explore, mode):
        count += 1
        game_over = False
        x = torch.from_numpy(self.cur_frame).cuda(0)
        logits, hidden = self.model(x.float(), prev.float(), e=min(max_drop, 1-completion_ratio), training=True)
        cpu_logits = logits.cpu()
        # if count % 50 == 0:
        #     print("move " + str(count) + " probs:  " + str(logits))
        epsilon = np.random.random(1)
        if epsilon <= (1-max_explore)*np.exp(.7*completion_ratio):
            if mode == 'det':
                move = torch.argmax(cpu_logits.data)
            elif mode == 'stoch':
                move = np.random.choice([0, 1, 2], p=cpu_logits.data.numpy().reshape(3))
            elif mode == 'threshold':
                if (max(cpu_logits.data)[0] < .5) and (min(cpu_logits.data)[0] > .2):
                    move = np.random.choice([0, 1, 2], p=cpu_logits.data.numpy().reshape(3))
                else:
                    move = torch.argmax(cpu_logits)
        else:
            # explore
            move = np.random.choice([0, 1, 2])
        if move == 1:
            if 1 == self.interface.E.move_left():
                game_over = True
        elif move == 2:
            if 1 == self.interface.E.move_right():
                game_over = True
        # if move is 1 go straight

        state, self.cur_frame = self.interface.update_board()
        if self.cur_frame is not None:
            self.cur_frame = self.cur_frame[4:46, :]

        # determine results of this move
        # Set Ys to penalize death, reward staying alive
        # Also weight loss more when dying, since this is a rare important event
        if state != 0 or game_over:
            coef = 1
            game_over = True
            y = torch.ones(3).cuda(0)
            y[move] = 0
            y = y.reshape(1, -1)
            #loss = (logits.pow(2)).sum() * nn.functional.binary_cross_entropy(logits, y)
            print(cpu_logits)
            print(loss)
        else:  # state == 0
            y = torch.ones(3).cuda(0)
            y[move] = 0
            y = y.reshape(1, -1)
            loss = .001*nn.functional.binary_cross_entropy(logits, y)*(logits.pow(2)).sum().pow(.5)

        # DO NOT zero grad. Allow time propagation
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if game_over:
            return state
        else:
            #time.sleep(.01)
            return self.recursive_train(hidden, completion_ratio, count, max_drop, max_explore, mode)

    def play_game(self, prev, count=0):
        game_over = False
        x = torch.from_numpy(self.cur_frame).cuda(0)
        logits, hidden = self.model(x.float(), prev.float(), e=.1, training=True)
        cpu_logits = logits.cpu()
        print("move " + str(count) + " probs:  " + str(logits))
        if (max(cpu_logits.data)[0] < .5) and (min(cpu_logits.data)[0] > .2):
            move = np.random.choice([0, 1, 2], p=cpu_logits.data.numpy().reshape(3))
        else:
            move = torch.argmax(cpu_logits)
        if move == 1:
            if 1 == self.interface.E.move_left():
                game_over = True
        elif move == 2:
            if 1 == self.interface.E.move_right():
                game_over = True
        # if move is 1 go straight
        state, self.cur_frame = self.interface.update_board()
        if self.cur_frame is not None:
            self.cur_frame = self.cur_frame[4:46, :]

        # determine results of this move
        # Set Ys to penalize death, reward staying alive
        # Also weight loss more when dying, since this is a rare important event
        if state != 0 or game_over:
            game_over = True
        if game_over:
            return state
        else:
            time.sleep(.01)
            return self.play_game(hidden, count=count+1)

    def eval(self, num_games):
        for i in range(num_games):
            self.interface = human_interface.Interface(human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)[4:46, :]
            self.optimizer.zero_grad()
            score = self.play_game(torch.zeros(50))
            print("score on game " + str(i) + ": " + str(score))


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    bob = Agent(lr=.001)
   # bob.model = torch.load('./models/trained1000.pkl')
   # bob.eval(10)
    bob.train(NUM_GAMES=1001, max_drop=.3, max_explore=.6, mode='threshold', disp_iter=100, save_iter=500)
    torch.cuda.device_count()



