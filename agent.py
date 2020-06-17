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
        self.model = ConvNetwork().float()#.cuda(0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.cur_frame = None
        self.short_term_memory = list()
        self.final_reward = int()

    def train(self, NUM_GAMES, max_drop=.5, max_explore=.4, disp_iter=10, save_iter=500):
        self.short_term_memory = list()
        self.final_reward = int()
        if max_drop > 1 or max_explore > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
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
            score = self.play_game(max_explore, i / NUM_GAMES, save_exp=True)
            self.meditate(.9)
            data_file.write(str(i) + ',' + str(score) + '\n')
            print("score on game " + str(i) + ": " + str(score))

    # initialize recursion
    def meditate(self, gamma):
        '''
        Preform batch graient decent on this play
        :param prev:
        :param completion_ratio:
        :param max_drop:
        :param max_explore:
        :param count:
        :return:
        '''
        num_states = len(self.short_term_memory)
        shape = self.short_term_memory[0][0].shape
        batch_x = torch.zeros(num_states, shape[0], shape[1])#.cuda(0)
        batch_y = torch.zeros(num_states)#.cuda(0)
        move_indices = torch.zeros(num_states).long()

        for i in range(num_states):
            batch_x[i] = self.short_term_memory[i][0]
            move_indices[i] = self.short_term_memory[i][1]
            batch_y[i] = (num_states - i) * np.power(gamma, i)
        # TODO: add gradients, clean, up, training logit
        batch_y_hat = self.model(batch_x, batch_size=num_states, training=False)

        self.optimizer.zero_grad()
        move_indices = move_indices.reshape([num_states, 1])
        batch_y_hat = batch_y_hat.gather(1, move_indices).reshape([-1]) #select moves that were made
        loss = nn.functional.mse_loss(batch_y_hat, batch_y)
        loss.backward()
        self.optimizer.step()

    def play_game(self, max_explore, completion_ratio, count=0, save_exp=True):
        game_over = False
        state = 0
        while not game_over:
            x = torch.from_numpy(self.cur_frame)#.cuda(0)
            exp_reward = self.model(x.float(), training=False)

            epsilon = np.random.random(1)
            if epsilon <= (1 - max_explore) * np.exp(.7 * completion_ratio):
                move = torch.argmax(exp_reward.data)
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

            if save_exp:
                self.short_term_memory.append([x.float(), torch.tensor(int(move)).long()])

            state, self.cur_frame = self.interface.update_board()
            if self.cur_frame is not None:
                self.cur_frame = self.cur_frame[4:46, :]
            if state > 0:
                game_over = True

        self.final_reward = state
        print("scored: " + str(state))
        return state

    def eval(self, num_games):
        for i in range(num_games):
            self.interface = human_interface.Interface(human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)[4:46, :]
            self.optimizer.zero_grad()
            score = self.play_game(0, 1, save_exp=False)
            print("score on game " + str(i) + ": " + str(score))


if __name__ == "__main__":
    #torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    bob = Agent(lr=.001)
   # bob.model = torch.load('./models/trained1000.pkl')
   # bob.eval(10)
    bob.train(NUM_GAMES=30, max_drop=.3, max_explore=.6, disp_iter=10, save_iter=10)
    #torch.cuda.device_count()



