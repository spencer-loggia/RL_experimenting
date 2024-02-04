import torch
from torch import nn
import numpy as np
from policy import ConvNetwork, SAPNet
import human_interface
import copy
import filters
import time
import sys
import pickle


class CogAgent:
    def __init__(self, env_type='grid_world', lr=.00001, dev='cpu', temporal_discount=.9, verbose=False):
        self.interface = None
        self.model = SAPNet(verbose=verbose)
        self.gradient_optimizer = torch.optim.SGD(self.model.parameters(), lr)
        self.cur_frame = None
        self.game_mode = 'grid_world'
        self.dev = dev
        self.gamma = temporal_discount
        self.actions = ['l', 'r', 'u', 'd', None]
        self.verbose = verbose

    def play_game(self, human_disp=False, epsilon=.6):
        loss_fxn = torch.nn.MSELoss()
        alive = True
        count = 0
        obj_count = 0
        if human_disp:
            self.interface = human_interface.Interface(human_player=False, human_disp=True,
                                                       game_mode='world', num_players=1,
                                                       grid_layout='data/layouts/10_10_maze.png')
        else:
            self.interface = human_interface.Interface(human_player=False, human_disp=False,
                                                       game_mode='world', num_players=1,
                                                       grid_layout='data/layouts/10_10_maze.png')

        self.cur_frame = self.interface.display_frame(pid=0)
        cur_reward_state = 0
        while cur_reward_state <= 0:
            self.gradient_optimizer.zero_grad()
            obs = filters.partial_observability_filter(self.cur_frame,
                                                       observe_dist=10,
                                                       origin=self.interface.E.cur_pos)
            height, width = obs.shape
            obs = obs.reshape((1, 1, height, width))

            exp_reward = self.model(obs, self.interface.E.hp)  # update voltages, encode, poll, and output
            if self.verbose:
                print('expected cost vector:', list(exp_reward))
            if np.nan in exp_reward:
                print("numerical error in expected reward computation", sys.stderr)
                raise ValueError

            if np.random.rand(1) < epsilon:
                action_code = np.random.choice(np.arange(4))
            else:
                action_code = torch.argmin(exp_reward)

            next_reward_state, next_frame = self.interface.update_board(self.actions[action_code], pid=0)
            if next_reward_state < 0:
                obj_count += 1
            if next_reward_state > 0:
                next_reward_hat = torch.Tensor([float(next_reward_state)])
            else:
                next_obs = filters.partial_observability_filter(next_frame, 10, self.interface.E.cur_pos)
                try:
                    next_obs = next_obs.reshape((1, 1, height, width))
                except Exception:
                    print('observation did not have requesite dimensionality', sys.stderr)
                    raise ValueError
                with torch.no_grad():
                    next_reward_hat = torch.min(self.model(next_obs, self.interface.E.hp))

            target = cur_reward_state + self.gamma * next_reward_hat
            if torch.isnan(target):
                print("numerical error in r state target computation", sys.stderr)
                raise ValueError
            pred = exp_reward[action_code]
            loss = loss_fxn(pred, target)
            if torch.isnan(loss):
                print("numerical error")
            loss.backward()
            self.gradient_optimizer.step()
            cur_reward_state = next_reward_state
            self.cur_frame = next_frame
            count += 1
        self.cur_frame = None
        self.interface = None
        self.gradient_optimizer.zero_grad()
        self.model.reset()
        print("lifetime: ", count, "rewards obtained: ", obj_count)
        return count

    def train(self, epochs=1001, max_explore=.85, disp_iter=100, save_iter=1000):
        lifetimes = []
        for i in range(epochs):
            completion_ratio = i / epochs
            epsilon = max(max_explore * (1 - completion_ratio), .02)
            if (i + 1) % save_iter == 0:
                file = open('./trained_models/' + self.game_mode + '_iter_' + str(i) + '.pkl',
                            'wb')
                pickle.dump(self.model, file)
            human_disp = False
            if ((i + 1) % disp_iter) == 0:
                print("DISPLAYING ITERATION", i, "epsilon:", epsilon)
                print(self.model.tdamn1.weight)
                print(self.model.tdamn2.weight)
                print(self.model.tdamn3.weight)
                human_disp = True
            count = self.play_game(human_disp, epsilon)
            lifetimes.append(count)
        return lifetimes


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
        self.cuda = True

    def train(self, NUM_GAMES, max_drop=.5, max_explore=.85, gamma=.9, disp_iter=100, save_iter=1000,
              game_mode='runner', trials_per_batch=100, lr=.001):
        if max_drop > 1 or max_explore > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
            return
        data_file = open('./data/' + game_mode + '_scores.csv', 'w')
        data_file.write("iter, score\n")
        self.optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=lr) for i in range(self.num_agents)]
        for i in range(int(NUM_GAMES / trials_per_batch)):

            # initialize new memory for batch
            self.short_term_memory = [
                [[] for j in range(trials_per_batch)]
                for i in range(self.num_agents)]
            self.reward_states = [
                [[] for j in range(trials_per_batch)]
                for i in range(self.num_agents)]

            for j in range(trials_per_batch):
                print("GAME # " + str(i))
                torch.cuda.empty_cache()
                self.final_reward = 0
                if (i + 1) * (j + 1) % save_iter == 0:
                    for w in range(self.num_agents):
                        file = open('./trained_models/' + game_mode + '_trained_agent_' + str(w) + '_iter_' + str(
                            (i + 1) * (j + 1)) + '.pkl',
                                    'wb')
                        torch.save(self.models[w], file)
                        file.close()
                if int((i + 1) * (j + 1) - 1) % disp_iter == 0:
                    if game_mode == 'runner':
                        self.interface = human_interface.Interface(human_player=False)
                    elif game_mode == 'snake':
                        self.interface = human_interface.Interface(human_player=False, game_mode='snake',
                                                                   num_players=self.num_agents, observe_dist=10)
                    elif game_mode == 'world':
                        self.interface = human_interface.Interface(human_player=False, game_mode='world',
                                                                   num_players=self.num_agents, observe_dist=10)
                else:
                    if game_mode == 'runner':
                        self.interface = human_interface.Interface(human_player=False, human_disp=False)
                    elif game_mode == 'snake':
                        self.interface = human_interface.Interface(human_player=False, human_disp=False,
                                                                   game_mode='snake', num_players=self.num_agents,
                                                                   observe_dist=10)
                    elif game_mode == 'world':
                        self.interface = human_interface.Interface(human_player=False, human_disp=False,
                                                                   game_mode='world', num_players=self.num_agents,
                                                                   observe_dist=10)
                for optimizer in self.optimizers: optimizer.zero_grad()

                self.play_game(max_explore, i / NUM_GAMES, trial_id=j, save_exp=True)

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
                agent_mem = [record[:, 0:2]]  # needs to be list to comply with meditation expectations
                agent_response = [record[:, 2]]  # needs to be list to comply with meditation expectations
                if lr != 0:
                    self.meditate(gamma, reward_dict, agent_mem, agent_response, self.models[0], self.optimizers[0])
        # set both trained_models to the supervised one
        for j in range(self.num_agents):
            self.models[j].load_state_dict(copy.deepcopy(self.models[0].state_dict()))
        file = open('./trained_models/' + game_mode + '_human_trained_iter_' + str(num_guided) + '.pkl', 'wb')
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
        shape = agent_mem[0][0][0].shape
        big_batch_x = torch.empty((0, shape[0], shape[1])).cuda(0)
        big_batch_y = torch.empty(0).cuda(0)
        big_batch_y_hat = torch.empty(0).cuda(0)
        for j in range(len(agent_mem)):
            num_states = len(agent_mem[j])
            batch_x = torch.zeros(num_states, shape[0], shape[1])  # .cuda(0)
            batch_y = torch.zeros(num_states)  # .cuda(0)
            move_indices = torch.zeros(num_states).long()  # .cuda(0)
            if self.cuda:
                batch_x = batch_x.cuda(0)
                batch_y = batch_y.cuda(0)
                move_indices = move_indices.cuda(0)
            # fill reward states
            for reward_type in list(reward_dict.keys()):
                inds = np.argwhere(np.array(agent_response[j]) == reward_type).reshape(-1)
                batch_y[inds] = reward_dict[reward_type]

            for i in range(num_states - 2, -1, -1):
                try:
                    batch_x[i] = torch.from_numpy(agent_mem[j, i, 0])
                except TypeError:
                    batch_x[i] = agent_mem[j][i][0]
                # recall what move was made to allow for partial gradient decent
                move_indices[i] = agent_mem[j][i][1]  # .cuda(0)
                # propogate rewards through time
                batch_y[i] = batch_y[i] + (gamma * batch_y[i + 1])
            batch_y_hat = model(batch_x.cuda(0), batch_size=num_states, training=False)
            optimizer.zero_grad()
            move_indices = move_indices.reshape([num_states, 1])
            batch_y_hat = batch_y_hat.gather(1, move_indices).reshape([-1]).cuda(0)  # select moves that were made
            big_batch_x = torch.cat((big_batch_x, batch_x), 0)
            big_batch_y = torch.cat([big_batch_y, batch_y], 0)
            big_batch_y_hat = torch.cat([big_batch_y_hat, batch_y_hat], 0)

        loss = nn.functional.mse_loss(big_batch_y_hat.cuda(0), big_batch_y.cuda(0))
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            pass

    def play_game(self, max_explore, completion_ratio, trial_id=0, save_exp=True, exploration_decay='linear'):
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
                exp_reward = self.models[i](x.float(),
                                            training=False)  # setting training to false to avoid use of dropout, which I found reduced
                # training speed w/o significant performance benefit in this case
                if count % 50 == 0:
                    print(exp_reward)
                # time.sleep(.1)
                t = 0
                if exploration_decay == 'exp':
                    t = (1 - max_explore) * np.exp(np.log(1 / (1 - max_explore)) * (1 / .9) * completion_ratio)
                elif exploration_decay == 'linear':
                    t = (1 - max_explore) + (max_explore * (completion_ratio - .1))

                epsilon = np.random.random(1)
                if epsilon <= t:
                    move = torch.argmax(exp_reward.data)
                else:
                    # explore
                    move = np.random.choice([0, 1, 2, 3])
                move = int(move)
                r, next_frame = self.interface.update_board(move_made=self.interface.action_code_map[move], pid=i)
                if next_frame is not None or False not in games_over[:i] + games_over[i + 1:]:
                    self.cur_frame = next_frame
                self.reward_states[i][trial_id].append(r)
                # if move is 4 or 5 go straight
                if r > 0:
                    games_over[i] = True
                if save_exp:
                    self.short_term_memory[i][trial_id].append([x.float(), torch.tensor(int(move)).long()])
            count += 1
        for i in range(self.num_agents):
            print("agent " + str(i) + " scored: " + str(self.interface.E.max_trail[i]))

    def load_models(self, iter_to_load, game_mode='snake', human_training=False):
        for i in range(self.num_agents):
            if not human_training:
                fname = './trained_models/' + game_mode + '_trained_agent_' + str(i) + '_iter_' + str(iter_to_load) + '.pkl'
            else:
                fname = './trained_models/' + game_mode + '_human_trained_iter_' + str(iter_to_load) + '.pkl'
            self.models[i] = torch.load(fname)



class ReverbAgent:

    def __init__(self):



if __name__ == "__main__":
    # torch.cuda.set_device(0)
    torch.autograd.set_detect_anomaly(True)
    bob = CogAgent(verbose=False)
    # bob.load_models(iter_to_load=10000, human_training=False)  # load saved policy to see trained agent play
    # bob.eval(10)
    # bob.learn_from_teacher(num_guided=4, lr=.01, game_mode='snake', gamma=.9)
    # bob.train(NUM_GAMES=1000001, max_explore=.9, trials_per_batch=100, lr=.001, gamma=.9, disp_iter=10000, save_iter=10000, game_mode='snake')
    bob.train(epochs=1000, max_explore=.9, disp_iter=50, save_iter=100)
    # torch.cuda.device_count()
