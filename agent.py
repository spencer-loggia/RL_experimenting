import numpy as np
import human_interface
import sys
import pandas as pd
import shutil
import h2oai_client
from scipy import ndimage
from PIL import Image

import os


def export_image_dataset(raw_data: (np.ndarray, np.ndarray, np.ndarray), batch_num: int, dai_data_dir: str, train=True) -> str:
    X = raw_data[0]
    moves = raw_data[1]
    Y = raw_data[2]
    try:
        os.mkdir(dai_data_dir + '/batch_' + str(batch_num))
    except FileExistsError:
        pass
    csv_str = 'StatePath,Action'
    if train:
        csv_str = csv_str + ',ObservedFutureReward'
    file = open(dai_data_dir + '/batch_' + str(batch_num) + '/labels.csv', 'w')
    for i in range(len(moves)):
        img = Image.fromarray(X[i].astype('uint8'), mode='L')
        im_dat = np.array(img)
        filename = 'sample_' + str(i) + '.png'
        img.save(dai_data_dir + '/batch_' + str(batch_num) + '/' + filename, format='PNG')
        csv_str = csv_str + '\n' + filename + ',' + str(int(moves[i]))
        if train:
            csv_str = csv_str + ',' + str(Y[i])
    file.write(csv_str)
    file.close()
    shutil.make_archive(dai_data_dir + '/sample_batch_' + str(batch_num), 'zip',
                        dai_data_dir + '/batch_' + str(batch_num))
    # shutil.rmtree(dai_data_dir + '/batch_' + str(batch_num))
    return '/data/batch_' + str(batch_num) + '.zip'


def export_image_pred_dataset(raw_data: (np.ndarray, np.ndarray), batch_num: int, dai_data_dir: str) -> str:
    X = raw_data[0]
    moves = raw_data[1]
    Y = raw_data[2]
    try:
        os.mkdir(dai_data_dir + '/batch_' + str(batch_num))
    except FileExistsError:
        pass
    csv_str = 'StatePath,Action'

    file = open(dai_data_dir + '/batch_' + str(batch_num) + '/labels.csv', 'w')
    for i in range(len(moves)):
        img = Image.fromarray(X[i].astype('uint8'), mode='L')
        im_dat = np.array(img)
        filename = 'sample_' + str(i) + '.png'
        img.save(dai_data_dir + '/batch_' + str(batch_num) + '/' + filename, format='PNG')
        csv_str = csv_str + '\n' + filename + ',' + str(int(moves[i]))

    file.write(csv_str)
    file.close()
    shutil.make_archive(dai_data_dir + '/sample_batch_' + str(batch_num), 'zip',
                        dai_data_dir + '/batch_' + str(batch_num))
    # shutil.rmtree(dai_data_dir + '/batch_' + str(batch_num))
    return '/data/sample-batch' + str(batch_num) + '.zip'


def export_csv_dataset(raw_data: np.ndarray, batch_num: int, dai_data_dir: str) -> str:
    try:
        os.mkdir(dai_data_dir + '/csv_data/')
    except FileExistsError:
        pass

    np.savetxt(dai_data_dir + '/csv_data/batch_' + str(batch_num) + 'data.csv', X=raw_data, delimiter=",")
    return dai_data_dir + '/csv_data/batch_' + str(batch_num) + 'data.csv'


class Agent:
    def __init__(self, username, password, url):
        self.interface = None
        self.dai = h2oai_client.Client(address=url, username=username, password=password)
        self.experiment = None
        self.cur_frame = None
        self.short_term_memory = list()

    def train(self, num_batches=4, batch_size=500, game_sample=10, epsilon=.5, gamma=.9, disp_iter=10,
              dai_data_dir='/home/sl/Projects/dai_rel_1.9_177/data'):
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
        :param dai_data_dir: directory to save training data in (absolute path)
        :return: None
        """
        if epsilon > 1:
            print('Invalid Hyperparameter. Learning Aborted', sys.stderr)
            return
        data_file = open('./data/scores.csv', 'w')
        data_file.write("iter,score\n")

        # main training_loop
        for i in range(num_batches):
            print("Beginning batch " + str(i) + " out of " + str(num_batches) + "....")
            self.play_all_games(batch_num=i, batch_size=batch_size, num_batches=num_batches, epsilon=epsilon)
            train_data = self.prepare_dataset(gamma, game_sample, flatten=False)
            zip_path = export_image_dataset(train_data, i, dai_data_dir)
            dai_data = self.dai.create_dataset_sync(zip_path)
            print('begining model update....')
            if self.experiment is None:
                self.experiment = self.dai.start_experiment_sync(dataset_key=dai_data.key,
                                                                 is_classification=False,
                                                                 target_col='ObservedFutureReward',
                                                                 is_image=True,
                                                                 accuracy=1,
                                                                 time=1,
                                                                 interpretability=8,
                                                                 scorer='RMSE')

            else:
                self.experiment = self.dai.start_experiment_sync(resumed_model_key=self.experiment.key,
                                                                 dataset_key=dai_data.key,
                                                                 is_classification=False,
                                                                 target_col='ObservedFutureReward',
                                                                 is_image=True,
                                                                 accuracy=1,
                                                                 time=1,
                                                                 interpretability=8,
                                                                 scorer='RMSE')

    def prepare_dataset(self, gamma, sample_size, flatten=False) -> (np.ndarray, np.ndarray, np.ndarray):
        DEATH_COST = -100
        STAY_ALIVE_REWARD = 1

        num_game_in_batch = len(self.short_term_memory)
        frame_dims = self.short_term_memory[0][0][0].shape  # batch 0, time step 0, states (not actions)

        # would use tensors but have to convert anyway eventually, no point in importing torch
        batch_x = np.zeros((0, frame_dims[0], frame_dims[1]))
        moves = np.zeros(0)
        batch_y = np.zeros(0)

        for j in range(num_game_in_batch):
            game_data = self.short_term_memory[j]
            # create state, action, reward sample indexes of desired size
            num_states = len(game_data)

            move_indices = np.zeros(num_states)
            temp_x = np.zeros((num_states, frame_dims[0], frame_dims[1]))
            temp_y = np.zeros(num_states)

            # compute q function on each time step
            final_move = num_states - 1
            temp_y[final_move] = DEATH_COST  # death penalized with -100
            temp_x[final_move] = game_data[final_move][0]
            move_indices[final_move] = game_data[final_move][1]

            # reverse through state action pairs computing true future reward recursively
            for i in range(num_states - 2, -1, -1):
                temp_x[i] = game_data[i][0]
                move_indices[i] = game_data[i][1]
                temp_y[i] = STAY_ALIVE_REWARD + (gamma * temp_y[i + 1])

            batch_x = np.concatenate([batch_x, temp_x], axis=0)
            moves = np.concatenate([moves, move_indices], axis=0)
            batch_y = np.concatenate([batch_y, temp_y], axis=0)

        if flatten:
            batch_y = batch_y.reshape(len(batch_y), 1)
            moves = moves.reshape(len(batch_y), 1)
            batch_x = batch_x.reshape(len(batch_y), -1)
            all_data = np.concatenate([batch_x, moves, batch_y], axis=1)
            return all_data
        print('generated training data...')
        return batch_x, moves, batch_y

    def get_next_move(self, action_set: list) -> list:
        '''
        really not ideal because of ridiculous HDD i/o
        :param action_set possible actions. must check pred for each since dai does not support n-dim regression
        :return:
        '''
        dim = self.cur_frame.shape
        X = self.cur_frame
        actions = []

        csv_str = 'StatePath,Action'

        dai_data_dir = '/data/tmp/'
        batch_num = -999

        try:
            os.mkdir(dai_data_dir + 'pred_batch_' + str(batch_num))
        except FileExistsError:
            pass

        file = open(dai_data_dir + 'pred_batch_' + str(batch_num) + '/labels.csv', 'w')
        for i in range(len(X)):
            img = Image.fromarray((X[i]*(255/2)).astype('uint8'), mode='L')
            filename = 'sample_' + str(i) + '.png'
            img.save(dai_data_dir + '/pred_batch_' + str(batch_num) + '/' + filename, format='PNG')
            for j in action_set:
                csv_str = csv_str + '\n' + filename + ',' + str(int(j))

        file.write(csv_str)
        file.close()
        shutil.make_archive(dai_data_dir + '/pred_batch_' + str(batch_num), 'zip',
                            dai_data_dir + '/pred_batch_' + str(batch_num))
        zip_path = dai_data_dir + '/pred_batch_' + str(batch_num) + '.zip'

        print('predicting')
        pred_data = self.dai.create_dataset_sync(zip_path)
        print('predicted')

        preds = self.dai.make_prediction_sync(model_key=self.experiment.key,
                                      dataset_key=pred_data.key,
                                      output_margin=False,
                                      pred_contribs=False,
                                      include_columns=['Action', 'ObservedFutureReward.predicted'])

        pred_path = self.dai.download(preds.predictions_csv_path, '/data/tmp/')
        pred_table = pd.read_csv(pred_path)
        pred_size = pred_table.shape[0]
        for i in range(0, pred_size, 3):
            one_game = pred_table[i:i+3]
            max_index = one_game['ObservedFutureReward'].argmax()
            actions.append(int(one_game['Action'].iloc[max_index]))
        return actions

    def play_all_games(self, batch_num: int, batch_size: int, num_batches: int, epsilon: float):
        all_over = False
        self.interface = [human_interface.Interface(human_disp=False, human_player=False) for i in range(batch_size)]
        completed_ids = [False]*batch_size
        self.short_term_memory = [list() for i in range(batch_size)]
        count = 0
        while not all_over:
            print("preforming action " + str(count) + " for " + str(batch_size) + " game instances")
            self.cur_frame = [np.zeros((50, 42)) for i in range(batch_size)]
            for j in range(batch_size):
                if not completed_ids[j]:
                    this_interface = self.interface[j]
                    if count == 0:
                        this_interface.update_board()
                    self.cur_frame[j] = np.array(this_interface.E.board_state)[4:46, :]

            self.cur_frame = np.array(self.cur_frame)
            if self.experiment is not None:
                moves = self.get_next_move([0, 1, 2])
            else:
                moves = np.empty(batch_size)
                epsilon = 1
            for j in range(batch_size):
                if not completed_ids[j]:
                    score = self.play_one_move(epsilon, batch_num / batch_size, j, moves[j], save_exp=True)
                    if score > 0:
                        completed_ids[j] = True
                        print("game " + str(batch_num*batch_size + j) + " score: " + str(score))
            if False not in completed_ids:
                all_over = True
            count += 1

    def play_one_move(self, max_explore, completion_ratio, game_index, move, save_exp=True):
        # TODO: Add change prediction to dai, need to predict for each action, choose best.
        game_over = False
        state = 0
        count = 0
        x = self.cur_frame[game_index]

        epsilon = np.random.random(1)
        if epsilon >= (1 - max_explore) * np.exp(.7 * completion_ratio) or self.experiment is None:
            # explore
            move = np.random.choice([0, 1, 2])
        if move == 1:
            if 1 == self.interface[game_index].E.move_left():
                game_over = True
        elif move == 2:
            if 1 == self.interface[game_index].E.move_right():
                game_over = True
        # if move is 1 go straight

        self.short_term_memory[game_index].append([x, int(move)])

        state, new_board = self.interface[game_index].update_board()
        if new_board is not None:
            self.cur_frame[game_index] = new_board[4:46, :]

        if state > 0:
            game_over = True
        count += 1
        return state

    def play_game(self, max_explore, completion_ratio, save_exp=True):
        # TODO: Add change prediction to dai, need to predict for each action, choose best.
        game_over = False
        state = 0
        count = 0
        while not game_over:
            x = self.cur_frame
            epsilon = np.random.random(1)
            if epsilon <= (1 - max_explore) * np.exp(.7 * completion_ratio) and self.experiment is not None:
                # predict for all possible moves
                move = self.get_next_move([0, 1, 2])
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
                self.short_term_memory[-1].append([x, int(move)])

            state, self.cur_frame = self.interface.update_board()
            if self.cur_frame is not None:
                self.cur_frame = self.cur_frame[4:46, :]
            if state > 0:
                game_over = True
            count += 1

        self.final_reward = state
        print("scored: " + str(state))
        return state

if __name__ == "__main__":
    bob = Agent(username='admin', password='admin', url='http://localhost:12345')
    # experiment_list = list(map(lambda x: x.key, bob.dai.list_models(offset=0, limit=100).models))
    # bob.experiment = bob.dai.get_model_job(experiment_list[0]).entity
    bob.train()
    # TODO: update main
