import math
import os.path
import random
from typing import List

import numpy as np
import torch
import pickle
import networkx as nx
from neurotools.modules import ElegantReverb
from neurotools.models import ElegantReverbNetwork

import filters
import human_interface
from scipy.ndimage import uniform_filter


def evo_step(life: List, exp: List, temporal_discount=.95):
    life_score = torch.zeros((1,))
    loss = torch.zeros((1,))
    exp.reverse()
    for i, instant_reward in enumerate(reversed(life)):
        life_score = life_score * temporal_discount + instant_reward
        loss = loss + torch.sqrt((life_score - exp[i]) ** 2)
    return loss


def clone(model: ElegantReverbNetwork):
    # wonkify starting edge weights
    for e in range(model.num_nodes):
        init_weight = model.edge.init_weight.detach().clone()
        if random.random() < 0:
            iw = init_weight[e]
            noise_var = random.random()
            noise_mean = random.random() - .5
            change = torch.normal(mean=noise_mean * .2, std=noise_var * .2, size=iw.shape)
            init_weight[e] = torch.nn.Parameter(init_weight[e].detach().clone() + change)

    # random ablate
    if random.random() < 0.00:
        mask = np.array(model.edge.mask)
        ind = np.random.choice(np.arange(mask.size))
        mask[ind // mask.shape[0], ind % mask.shape[0]] = 0
        mask = torch.from_numpy(mask)
        model.edge.mask = mask

    # tonic noise
    if random.random() < 0:
        model.noise += abs(random.random() * .3 - .15)

    # resistance
    if random.random() < .00:
        model.resistance += random.random() * .3 - .15

    return model.detach(reset_intrinsic=True)


class MetaAgent:
    def __init__(self, lr=.001, dev='cpu', temporal_discount=.9, verbose=False, num_nodes=4, spatial=5):
        self.epsilon_min = .01
        self.epsilon_slope = .00002
        self.epsilon_init = .5
        self.interface = None
        self.spatial = 5
        self.start_nodes = 3
        self.input_node = -1
        self.decode_node = 2
        self.num_nodes = num_nodes
        self.reward_node = 3
        self.lr = lr
        adj = torch.zeros((num_nodes, num_nodes))
        adj[:, 2] = 1
        adj[2, :] = 1
        self.model = ElegantReverbNetwork(num_nodes=num_nodes, node_shape=(1, 3, spatial, spatial), kernel_size=4,
                                          edge_module=ElegantReverb, track_activation_history=True, mask=adj)
        self.stim_frames = 1
        self.actions = ['l', 'r', 'u', 'd', None]
        self.action_decoder = torch.nn.Linear(in_features=self.spatial**2, out_features=5)
        self.gradient_optimizer = None
        self.cur_frame = None
        self.game_mode = 'grid_world'
        self.dev = dev
        self.gamma = temporal_discount

        self.verbose = verbose

    def step(self, generation):
        obs = filters.partial_observability_filter(self.cur_frame,
                                                   observe_dist=math.floor(self.spatial / 2),
                                                   origin=self.interface.E.cur_pos).squeeze()
        temp = torch.zeros_like(self.model.states)
        temp[0, 0, :, :] += obs
        obs = temp.detach().clone()
        for _ in range(self.stim_frames):
            out_state = self.model(obs)[1, 0, :, :]
        action_vec = self.action_decoder(out_state.reshape(1, -1)).flatten()
        eps = max(self.epsilon_init + -1 * self.epsilon_slope * generation, self.epsilon_min)
        draw = random.random()
        if draw >= eps:
            action = torch.argmax(action_vec)
        else:
            action = random.randint(0, 4)
        exp_reward = action_vec[action]
        return action, exp_reward

    def exist(self, generation: int, env_layout='data/layouts/10_10_maze.png', human_disp=False):
        if human_disp:
            self.interface = human_interface.Interface(human_player=False, human_disp=True,
                                                       game_mode='world', num_players=1,
                                                       grid_layout=env_layout)
        else:
            self.interface = human_interface.Interface(human_player=False, human_disp=False,
                                                       game_mode='world', num_players=1,
                                                       grid_layout=env_layout)
        # clone for next generations
        self.model = clone(self.model)
        self.gradient_optimizer = torch.optim.SGD(list(self.model.parameters()) +
                                                   list(self.action_decoder.parameters()),
                                                   self.lr*(.99985**generation))
        self.gradient_optimizer.zero_grad()
        life_history = []
        exp_reward_history = []
        self.cur_frame = self.interface.display_frame(pid=0)
        reward = 0
        span = 0
        while reward >= 0:
            action, exp_reward = self.step(generation)
            exp_reward_history.append(exp_reward)
            reward, next_frame = self.interface.update_board(self.actions[action], pid=0)
            life_history.append(reward)
            self.cur_frame = next_frame
            span += 1
        return life_history, exp_reward_history, span

    def evolve(self, generations, disp_iter=5000, snapshot_iter=5000, model_dir='./'):
        loss_history = []
        lifespan_history = []
        for generation in range(generations):
            print("GENERATION", generation, "OF", generations)
            if (generation % disp_iter) == 0:
                life, exp, span = self.exist(generation, human_disp=True)
            else:
                life, exp, span = self.exist(generation, human_disp=False)
            print("Survived", span, "steps")
            lifespan_history.append(span)
            loss = evo_step(life, exp, temporal_discount=.95)
            loss_history.append(loss.detach().cpu().item() / span)
            print("Agent loss:", loss_history[-1], "\n")
            if ((generation + 1) % snapshot_iter) == 0:
                with open(os.path.join(model_dir, "reverb_snapshot_" + str(generation) + ".pkl"), "wb") as f:
                    pickle.dump(self, f)
            loss.backward(create_graph=False)
            self.gradient_optimizer.step()
        return loss_history, lifespan_history


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    agent = MetaAgent()
    evolutionary_history, lifespan_history = agent.evolve(100000, model_dir="trained_models")
    plt.plot(uniform_filter(np.array(lifespan_history), 100))
    plt.show()
