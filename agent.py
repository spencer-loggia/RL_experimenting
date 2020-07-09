import numpy as np
import human_interface
import sys
import shutil
from PIL import Image
import driverlessai
import os


def export_dataset(raw_data: (np.ndarray, np.ndarray, np.ndarray), batch_num: int) -> str:
    X = raw_data[0]
    moves = raw_data[1]
    Y = raw_data[2]
    os.mkdir('data/batch_' + str(batch_num))
    csv_str = 'StatePath,Action,ObservedFutureReward'
    file = open('data/batch_' + str(batch_num) + '/labels.csv', 'w')
    for i in range(len(Y)):
        img = Image.fromarray(X[i])
        filename = 'sample_' + str(i) + '.jpeg'
        img.save('data/batch_' + str(batch_num) + '/' + filename)
        csv_str = csv_str + '\n' + filename + ',' + str(moves[i]) + ',' + str(Y[i])
    file.write(csv_str)
    file.close()
    shutil.make_archive('data/sample_batch_' + str(batch_num) + '.zip', 'zip', 'data/batch_' + str(batch_num))
    os.rmdir('data/sample_batch_' + str(batch_num))
    return 'data/sample_batch_' + str(batch_num) + '.zip'

class Agent:
    def __init__(self, username, password, url):
        self.interface = None
        self.dai = driverlessai.Client(address=url, username=username, password=password)
        self.experiment = None
        self.cur_frame = None
        self.short_term_memory = list()
        self.final_reward = int()

    def train(self, num_batches=100, batch_size=100, game_sample=100, eplison=.4, gamma=.9, disp_iter=10, save_iter=500):
        """
        Main training function
        :param num_batches: number of training iterations, each with number of games equal to batch size.
        :param batch_size: the number of games before model is updated
        :param game_sample: number of time steps to train on from each game
        :param eplison: the starting probability that a random action is taken (epsilon-greedy). Decays
        exponentially with number of games
        :param gamma: discounting factor, the base of exponential decay of future event rewards
        :param disp_iter: how frequently (in games) a visualization of agent's game play should be displayed
        (slows training)
        :return: None
        """
        if eplison > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
            return
        data_file = open('./data/scores.csv', 'w')
        data_file.write("iter,score\n")

        # main training_loop
        for i in range(num_batches):
            self.short_term_memory = []
            for j in range(batch_size):
                self.final_reward = 0
                if i % disp_iter == 0:
                    self.interface = human_interface.Interface(human_player=False)
                else:
                    self.interface = human_interface.Interface(human_disp=False, human_player=False)
                self.cur_frame = np.array(self.interface.E.board_state)[4:46, :]
                score = self.play_game(eplison, (i * j) / (num_batches * batch_size), save_exp=True)
            train_data = self.prepare_dataset(gamma, game_sample)
            zip_path = export_dataset(train_data, i)
            dai_data = self.dai.datasets.create(data=zip_path, data_source='upload', name='RL_frame_' + str(i))
            if self.experiment is None:
                self.experiment = self.dai.experiments.create(name='RL_test',
                                                              test_dataset=dai_data,
                                                              task='regression',
                                                              target_column='ObservedFutureReward',
                                                              accuracy=5,
                                                              time=3,
                                                              interpretability=5)
            else:
                self.experiment.retrain(use_smart_checkpoint=True,
                                        test_dataset=dai_data,
                                        task='regression',
                                        target_column='ObservedFutureReward',
                                        accuracy=5,
                                        time=3,
                                        interpretability=5)


    def create_scoring_pipeline(self):
        self.experiment

    def prepare_dataset(self, gamma, sample_size)-> (np.ndarray, np.ndarray, np.ndarray):
        DEATH_COST = -100
        STAY_ALIVE_REWARD = 1

        num_game_in_batch = len(self.short_term_memory)
        frame_dims = self.short_term_memory[0][0].shape

        # would use tensors but have to convert anyway eventually, no point in importing torch
        batch_x = np.zeros((num_game_in_batch, sample_size, frame_dims[0], frame_dims[1]))
        moves = np.zeros((num_game_in_batch, sample_size))
        batch_y = np.zeros((num_game_in_batch, sample_size))

        for j in range(num_game_in_batch):
            game_data = self.short_term_memory[j]
            # create state, action, reward sample indexes of desired size
            num_states = len(game_data)
            sample_indexes = np.random.choice(num_states, sample_size)

            move_indices = np.zeros(num_states)

            # compute q function on each time step
            final_move = num_states - 1
            batch_y[final_move] = DEATH_COST  # death penalized with -100
            batch_x[final_move] = game_data[final_move][0]
            move_indices[final_move] = game_data[final_move][1]

            temp_x = np.zeros(num_states, frame_dims[0], frame_dims[1])
            temp_y = np.zeros(num_states)
            # reverse through state action pairs computing true future reward recursively
            for i in range(num_states - 2, -1, -1):
                temp_x[i] = self.short_term_memory[i][0]
                move_indices[i] = game_data[i][1]
                temp_y[j][i] = STAY_ALIVE_REWARD + (gamma * batch_y[i + 1])

            # sample
            batch_x[j] = temp_x[sample_indexes]
            moves[j] = move_indices[sample_indexes]
            batch_y[j] = temp_y[sample_indexes]

        # reduce dimensionality to fit dai label conventions
        batch_x = batch_x.reshape([num_game_in_batch * sample_size, frame_dims[0], frame_dims[1]])
        moves = moves.reshape([num_game_in_batch * sample_size])
        batch_y = batch_y.reshape([num_game_in_batch * sample_size])
        return batch_x, moves, batch_y

    def play_game(self, max_explore, completion_ratio, save_exp=True):
        # TODO: Add change prediction to dai, need to predict for each action, choose best.
        game_over = False
        state = 0
        count = 0
        while not game_over:
            x = self.cur_frame
            exp_reward = self.model(x.float(), training=False)

            if count % 10 == 0:
                print(exp_reward)

            epsilon = np.random.random(1)
            if epsilon <= (1 - max_explore) * np.exp(.7 * completion_ratio):
                # predict for all possible moves
                move = np.argmax(exp_reward.data)
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
                self.short_term_memory.append([x.float(), int(move)])

            state, self.cur_frame = self.interface.update_board()
            if self.cur_frame is not None:
                self.cur_frame = self.cur_frame[4:46, :]
            if state > 0:
                game_over = True
            count += 1

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
    pass
    #TODO: update main
