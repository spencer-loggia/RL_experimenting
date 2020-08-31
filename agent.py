import torch
from torch import nn
import numpy as np
from policy import ConvNetwork
import human_interface
import copy
import filters
import time
import sys


class Agent:
    def __init__(self, num_agents=1, partial_env=False):
        self.interface = None
        self.models = [ConvNetwork().float() for i in range(num_agents)]  # .cuda(0)
        self.optimizers = None
        self.cur_frame = None
        self.short_term_memory = None
        self.num_agents = num_agents
        self.final_reward = int()
        self.reward_states = None
        self.game_mode = None

    def train(self, NUM_GAMES, max_drop=.5, max_explore=.85, gamma=.9, disp_iter=100, save_iter=1000,
              game_mode='runner', lr=.001):
        if max_drop > 1 or max_explore > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
            return
        data_file = open('./data/' + game_mode + '_scores.csv', 'w')
        data_file.write("iter, score\n")
        self.optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=lr) for i in range(self.num_agents)]
        for i in range(NUM_GAMES):
            print("GAME # " + str(i))
            torch.cuda.empty_cache()
            self.short_term_memory = [[] for i in range(self.num_agents)]
            self.reward_states = [[] for i in range(self.num_agents)]
            self.final_reward = 0
            if (i + 1) % save_iter == 0:
                for j in range(self.num_agents):
                    file = open('./models/' + game_mode + '_trained_agent_' + str(j) + '_iter_' + str(i + 1) + '.pkl', 'wb')
                    torch.save(self.models[j], file)
                    file.close()
            if i % disp_iter == 0:
                if game_mode == 'runner':
                    self.interface = human_interface.Interface(human_player=False)
                elif game_mode == 'snake':
                    self.interface = human_interface.Interface(human_player=False, game_mode='snake',
                                                               num_players=self.num_agents, observe_dist=10)
            else:
                if game_mode == 'runner':
                    self.interface = human_interface.Interface(human_player=False, human_disp=False)
                elif game_mode == 'snake':
                    self.interface = human_interface.Interface(human_player=False, human_disp=False,
                                                               game_mode='snake', num_players=self.num_agents,
                                                               observe_dist=10)
            for optimizer in self.optimizers: optimizer.zero_grad()

            self.play_game(max_explore, i / NUM_GAMES, save_exp=True)

            # set reward dict
            reward_dict = None
            if game_mode == 'runner':
                reward_dict = {0: 1, 1: -100}
            elif game_mode == 'snake':
                reward_dict = {0: 0, -1: 1, 1: -1}
            if lr != 0:
                for i in range(self.num_agents):
                    self.meditate(gamma, reward_dict,
                                  agent_mem=self.short_term_memory[i],
                                  agent_response=self.reward_states[i],
                                  model=self.models[i],
                                  optimizer=self.optimizers[i])

    def learn_from_teacher(self, num_guided=10, game_mode='snake', gamma=.9, lr=.01, load_preplayed=False):
        # assuming from scratch for now, overwrites existing model
        # set reward dict
        reward_dict = None
        self.optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=lr) for i in range(self.num_agents)]
        if game_mode == 'runner':
            reward_dict = {0: 1, 1: -100}
        elif game_mode == 'snake':
            reward_dict = {0: 0, -1: 1, 1: -1}
        for i in range(num_guided):
            if not load_preplayed:
                self.interface = human_interface.Interface(human_player=True, human_disp=True, record_game=True,
                                                           game_mode=game_mode, num_players=1,
                                                           observe_dist=10)
                self.interface.game_loop()
                record = np.array(self.interface.record)
                # determine moves made by model
                agent_mem = record[:, 0:2]
                agent_response = record[:, 2]
                if lr != 0:
                    self.meditate(gamma, reward_dict, agent_mem, agent_response, self.models[0], self.optimizers[0])
        # set both models to the supervised one
        for j in range(self.num_agents):
            self.models[j].load_state_dict(copy.deepcopy(self.models[0].state_dict()))
        file = open('./models/' + game_mode + '_human_trained_iter_' + str(num_guided) + '.pkl', 'wb')
        torch.save(self.models[0], file)

    def meditate(self, gamma, reward_dict, agent_mem, agent_response, model, optimizer):
        """
         Learn from previous game
        :param gamma: the time discounting factor
        :param reward_dict: the mapping of response states to reward values. generally negative response has positive
        :param agent_mem: the
        reward and positive response has negative reward
        :return: None
        """
        num_states = len(agent_mem)
        shape = agent_mem[0][0].shape
        batch_x = torch.zeros(num_states, shape[0], shape[1])  # .cuda(0)
        batch_y = torch.zeros(num_states)  # .cuda(0)
        move_indices = torch.zeros(num_states).long()  # .cuda(0)
        # fill reward states
        for reward_type in list(reward_dict.keys()):
            inds = np.argwhere(np.array(agent_response) == reward_type).reshape(-1)
            batch_y[inds] = reward_dict[reward_type]

        for i in range(num_states - 2, -1, -1):
            try:
                batch_x[i] = torch.from_numpy(agent_mem[i, 0])
            except TypeError:
                batch_x[i] = agent_mem[i][0]
            # recall what move was made to allow for partial gradient decent
            move_indices[i] = agent_mem[i][1]  # .cuda(0)
            # propogate rewards through time
            batch_y[i] = batch_y[i] + (gamma * batch_y[i + 1])
        batch_y_hat = model(batch_x, batch_size=num_states, training=False)
        optimizer.zero_grad()
        move_indices = move_indices.reshape([num_states, 1])
        batch_y_hat = batch_y_hat.gather(1, move_indices).reshape([-1])  # select moves that were made
        loss = nn.functional.mse_loss(batch_y_hat, batch_y)
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            pass

    def play_game(self, max_explore, completion_ratio, save_exp=True, exploration_decay='linear'):
        count = 0
        games_over = [False for i in range(self.num_agents)]
        # need to preform each agents move in parallel (if possible) on each game step.
        # it is neccesary to randomize which agent goes first in order to make up for hardware
        # discrepencies if parallel and natural ordering if not.
        order = np.array(list(range(self.num_agents)))
        while False in games_over:
            np.random.shuffle(order)
            for i in order:
                if games_over[i]:
                    continue
                if self.cur_frame is None and not games_over[i]:
                    self.cur_frame = self.interface.display_frame(pid=i)
                x = torch.from_numpy(self.cur_frame)
                if x.shape[0] != 20:
                    print('ruh-ro')
                exp_reward = self.models[i](x.float(), training=False)  # setting training to false to avoid use of dropout, which I found reduced
                                                                    # training speed w/o significant performance benefit in this case
                if count % 50 == 0:
                    print(exp_reward)
                # time.sleep(.1)
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
                r, next_frame = self.interface.update_board(move_made=self.interface.action_code_map[move], pid=i)
                if next_frame is not None or False not in games_over[:i] + games_over[i+1:]:
                    self.cur_frame = next_frame
                self.reward_states[i].append(r)
                # if move is 4 or 5 go straight
                if r > 0:
                    games_over[i] = True
                if save_exp:
                    self.short_term_memory[i].append([x.float(), torch.tensor(int(move)).long()])
            count += 1
        for i in range(self.num_agents):
            print("agent " + str(i) + " scored: " + str(self.interface.E.max_trail[i]))

    def load_models(self, iter_to_load, game_mode='snake', human_training=False):
        for i in range(self.num_agents):
            if not human_training:
                fname = './models/' + game_mode + '_trained_agent_' + str(i) + '_iter_' + str(iter_to_load) + '.pkl'
            else:
                fname = './models/' + game_mode + '_human_trained_iter_' + str(iter_to_load) + '.pkl'
            self.models[i] = torch.load(fname)


if __name__ == "__main__":
    # torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    bob = Agent(num_agents=1)
    bob.load_models(iter_to_load=4, human_training=True)  # load saved policy to see trained agent play
    # bob.eval(10)
    #bob.learn_from_teacher(num_guided=4, lr=.01, game_mode='snake', gamma=.9)
    bob.train(NUM_GAMES=1001, max_explore=.8, lr=.001, gamma=.9, disp_iter=100, save_iter=1000, game_mode='snake')
    #bob.train(NUM_GAMES=10, max_explore=0, lr=0, gamma=.9, disp_iter=1, save_iter=1000, game_mode='snake')
    # torch.cuda.device_count()
