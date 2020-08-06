import torch
from torch import nn
import numpy as np
from policy import ConvNetwork
import human_interface
import time
import sys


class Agent:
    def __init__(self, lr=.001):
        self.interface = None
        self.model = ConvNetwork().float()  # .cuda(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cur_frame = None
        self.short_term_memory = list()
        self.final_reward = int()
        self.reward_states = []  # list of tuples of len num states in game, (state, reward)

    def train(self, NUM_GAMES, max_drop=.5, max_explore=.85, gamma=.9, disp_iter=100, save_iter=1000,
              game_mode='runner'):
        if max_drop > 1 or max_explore > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
            return
        data_file = open('./data/' + game_mode + '_scores.csv', 'w')
        data_file.write("iter,score\n")
        self.model = self.model.float()
        for i in range(NUM_GAMES):
            torch.cuda.empty_cache()
            self.short_term_memory = []
            self.reward_states = []
            self.final_reward = 0
            if (i + 1) % save_iter == 0:
                file = open('./models/' + game_mode + '_trained' + str(i + 1) + '.pkl', 'wb')
                torch.save(self.model, file)
                file.close()
            if i % disp_iter == 0:
                if game_mode == 'runner':
                    self.interface = human_interface.Interface(human_player=False)
                elif game_mode == 'snake':
                    self.interface = human_interface.Interface(human_player=False, game_mode='snake')
            else:
                if game_mode == 'runner':
                    self.interface = human_interface.Interface(human_player=False, human_disp=False)
                elif game_mode == 'snake':
                    self.interface = human_interface.Interface(human_player=False, human_disp=False, game_mode='snake')
            self.cur_frame = np.array(self.interface.E.board_state)
            self.optimizer.zero_grad()
            score = self.play_game(max_explore, i / NUM_GAMES, save_exp=True)

            # set reward dict
            reward_dict = None
            if game_mode == 'runner':
                reward_dict = {0: 1, 1: -100}
            elif game_mode == 'snake':
                reward_dict = {0: 0, -1: 1, 1: -1}
            self.meditate(gamma, reward_dict)
            data_file.write(str(i) + ',' + str(score) + '\n')
            print("score on game " + str(i) + ": " + str(score))

    # initialize recursion
    def meditate(self, gamma, reward_dict):
        """
        Learn from previous game
        :param gamma: the time discounting factor
        :param reward_dict: the mapping of response states to reward values. generally negative response has positive
        reward and positive response has negative reward
        :return: None
        """
        num_states = len(self.short_term_memory)
        if num_states > 3250:
            # stop-gap to prevent memory overflow, play basically optimal at this point anyway
            # better way to do this would be to base cut off on available system mem
            return
        shape = self.short_term_memory[0][0].shape
        batch_x = torch.zeros(num_states, shape[0], shape[1])  # .cuda(0)
        batch_y = torch.zeros(num_states)  # .cuda(0)
        move_indices = torch.zeros(num_states).long()  # .cuda(0)
        # fill reward states
        for reward_type in list(reward_dict.keys()):
            inds = np.argwhere(np.array(self.reward_states) == reward_type).reshape(-1)
            batch_y[inds] = reward_dict[reward_type]

        for i in range(num_states - 2, -1, -1):
            batch_x[i] = self.short_term_memory[i][0]
            # recall what move was made to allow for partial gradient decent
            move_indices[i] = self.short_term_memory[i][1]  # .cuda(0)
            # propogate rewards through time
            batch_y[i] = batch_y[i] + (gamma * batch_y[i + 1])

        batch_y_hat = self.model(batch_x, batch_size=num_states, training=False)
        self.optimizer.zero_grad()
        move_indices = move_indices.reshape([num_states, 1])
        batch_y_hat = batch_y_hat.gather(1, move_indices).reshape([-1])  # select moves that were made
        loss = nn.functional.mse_loss(batch_y_hat, batch_y)
        try:
            loss.backward()
            self.optimizer.step()
        except RuntimeError:
            pass

    def play_game(self, max_explore, completion_ratio, save_exp=True, exploration_decay='linear'):
        game_over = False
        state = 0
        count = 0
        response = 0
        while not game_over:
            x = torch.from_numpy(self.cur_frame)  # .cuda(0)
            exp_reward = self.model(x.float(),
                                    training=False)  # setting training to false to avoid use of dropout, which I found reduced
            # training speed w/o significant preformance benefit in this case
            if count % 10 == 0:
                print(exp_reward)
            #time.sleep(.1)

            t = 0
            if exploration_decay == 'exp':
                t = (1 - max_explore) * np.exp(np.log(1 / (1 - max_explore)) * (1 / .9) * completion_ratio)
            elif exploration_decay == 'linear':
                t = (1 - max_explore) + (max_explore * (completion_ratio-.1))

            epsilon = np.random.random(1)
            if epsilon <= t:
                move = torch.argmax(exp_reward.data)
            else:
                # explore
                move = np.random.choice([0, 1, 2, 3])
            move = int(move)
            response = 0
            if move == 0:
                response, self.cur_frame = self.interface.update_board(move_made='l')
            elif move == 1:
                response, self.cur_frame = self.interface.update_board(move_made='r')
            elif move == 2:
                response, self.cur_frame = self.interface.update_board(move_made='u')
            elif move == 3:
                response, self.cur_frame = self.interface.update_board(move_made='d')
            else:
                response, self.cur_frame = self.interface.update_board()

            if 1 <= response:
                game_over = True
            # if move is 4 or 5 go straight

            if save_exp:
                self.short_term_memory.append([x.float(), torch.tensor(int(move)).long()])
                self.reward_states.append(response)
            count += 1

        print("scored: " + str(self.interface.E.max_trail))
        return response

    # TODO: impliment working play back / evaluation (below is non functional hold over)
    # Currently can see agent perform once trained by setting lr=0 and display_iter=1 and max_explore=0
    def eval(self, num_games):
        for i in range(num_games):
            self.interface = human_interface.Interface(human_player=False)
            self.cur_frame = np.array(self.interface.E.board_state)
            self.optimizer.zero_grad()
            score = self.play_game(0, 1, save_exp=False)
            print("score on game " + str(i) + ": " + str(score))


if __name__ == "__main__":
    # torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    bob = Agent(lr=.001)
    #bob.model = torch.load('./models/snake_trained10000.pkl')  # load saved policy to see trained agent play
    # bob.eval(10)
    bob.train(NUM_GAMES=1000001, max_explore=1, gamma=.9, disp_iter=1000, save_iter=10000, game_mode='snake')
    # torch.cuda.device_count()
