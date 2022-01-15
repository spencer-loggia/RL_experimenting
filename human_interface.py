import tkinter as tk
from tkinter import TclError
import eviroment
import filters
import numpy as np
from tkinter import Tk, Canvas, Frame, Text, INSERT
from PIL import Image, ImageTk
import scipy.ndimage
import time


class Interface:
    def __init__(self, human_disp=True, human_player=True, record_game=False, game_mode='runner', num_players=1, observe_dist=None, FPS=10, **kwargs):
        self.game_mode = game_mode
        self.action_code_map = None
        if game_mode == 'runner':
            self.E = eviroment.RunnerEnv()
        elif game_mode == 'snake':
            self.action_code_map = ['l', 'r', 'u', 'd']
            self.E = eviroment.SnakeEnv(num_player=num_players)
        elif game_mode == 'world':
            self.action_code_map = ['l', 'r', 'u', 'd', None]
            self.E = eviroment.GridWorldRevolution(input_layout=kwargs['grid_layout'])
        self.raw_state = self.E.board_state
        self.human_disp = human_disp
        self.human_player = human_player
        self.command = ''
        self.root = None
        self.observe_dist = observe_dist
        self.record = None
        if human_disp:
            self.root = Tk()
            board_img = np.array(self.raw_state) * int(255 / 2)
            np_img = scipy.ndimage.zoom(board_img, 8, order=0)
            img = ImageTk.PhotoImage(image=Image.fromarray(np_img), master=self.root)
            self.canvas = Canvas(self.root, width=np_img.shape[1], height=np_img.shape[0])
            self.canvas.create_image(1, 2, anchor="nw", image=img)
            self.canvas.pack()
            self.canvas.update_idletasks()
            self.canvas.update()
        if human_player:
            self.frame = Frame(self.root, width=100, height=100)
            self.root.bind('<Left>', self.leftKey)
            self.root.bind('<Right>', self.rightKey)
            self.root.bind('<Up>', self.upKey)
            self.root.bind('<Down>', self.downKey)
            self.frame.pack()
            if record_game:
                self.record = []

    def leftKey(self, event):
        if self.human_player:
            if self.E.move('l') == 1 and self.root is not None:
                #self.canvas.destroy()
                self.root.destroy()

    def rightKey(self, event):
        if self.human_player:
            if self.E.move('r') == 1 and self.root is not None:
                #self.canvas.destroy()
                self.root.destroy()

    def upKey(self, event):
        if self.human_player:
            if self.E.move('u') == 1 and self.root is not None:
                # self.canvas.destroy()
                self.root.destroy()

    def downKey(self, event):
        if self.human_player:
            if self.E.move('d') == 1 and self.root is not None:
                # self.canvas.destroy()
                self.root.destroy()

    def display_frame(self, toContinue=True, pid=0):
        if toContinue:
            board_img = np.array(self.raw_state)
            if self.observe_dist is not None:
                cur_pos = tuple(self.E.cur_pos[pid])
                x = filters.partial_observability_filter(board_img, self.observe_dist, cur_pos)
            else:
                x = board_img  # .cuda(0)
            if self.human_disp:
                time.sleep(.05)
                try:
                    self.canvas.delete('all')
                    np_img = scipy.ndimage.zoom(board_img * int(255 / 2), 8, order=0)
                    img = ImageTk.PhotoImage(image=Image.fromarray(np_img), master=self.root)
                    self.canvas.create_image(1, 2, anchor="nw", image=img)
                    self.canvas.pack()
                    self.root.update_idletasks()
                    self.root.update()
                except tk.TclError:
                    if self.root is not None:
                        self.root = None
            return x
        else:
            print("Game Over. Score: " + str(self.E.line_count))
        return None

    def update_board(self, move_made=None, pid=0):
        win_state = 0
        if move_made is None:
            win_state = self.E.move(None, pid=pid)
        elif move_made == 'l':
            win_state = self.E.move('l', pid=pid)
        elif move_made == 'r':
            win_state = self.E.move('r', pid=pid)
        elif move_made == 'u':
            win_state = self.E.move('u', pid=pid)
        elif move_made == 'd':
            win_state = self.E.move('d', pid=pid)

        if move_made is None and self.game_mode == 'snake':
            code = self.action_code_map.index(self.E.cur_direction[pid])
        else:
            code = self.action_code_map.index(move_made)
        self.raw_state = self.E.board_state

        cur_frame = self.display_frame(pid=pid)
        if self.record is not None:
            self.record.append([cur_frame, code, win_state])
        if win_state > 0:
            cur_frame = None
            if self.root is not None and self.E.num_alive == 0:
                try:
                    self.root.destroy()
                except TclError:
                    pass
        return win_state, cur_frame

    def game_loop(self):
        state = 0
        while state <= 0:
            time.sleep(.075)
            state, cur_frame = self.update_board()
        return state


if __name__ == "__main__":
    test = Interface(game_mode='world', grid_layout='data/layouts/10_10_maze.png')
    print("you scored:" + str(test.game_loop()))




