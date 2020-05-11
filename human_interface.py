import tkinter as tk
import eviroment
import numpy as np
from tkinter import Tk, Canvas, Frame, Text, INSERT
from PIL import Image, ImageTk
import scipy.ndimage
import time

class Interface:
    def __init__(self, human_disp=True, human_player=True, FPS=10):
        self.E = eviroment.Env()
        self.raw_state = self.E.board_state
        self.human_disp = human_disp
        self.human_player = human_player
        self.command = ''
        self.root = None
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
            self.frame.pack()

    def leftKey(self, event):
        if self.human_player:
            if self.E.move_left() == 1 and self.root is not None:
                #self.canvas.destroy()
                self.root.destroy()

    def rightKey(self, event):
        if self.human_player:
            if self.E.move_right() == 1 and self.root is not None:
                #self.canvas.destroy()
                self.root.destroy()

    def display_frame(self, toContinue=True):
        if toContinue:
            board_img = np.array(self.raw_state) * int(255 / 2)
            if self.human_disp:
                try:
                    self.canvas.delete('all')
                    np_img = scipy.ndimage.zoom(board_img, 8, order=0)
                    img = ImageTk.PhotoImage(image=Image.fromarray(np_img), master=self.root)
                    self.canvas.create_image(1, 2, anchor="nw", image=img)
                    self.canvas.pack()
                    self.root.update_idletasks()
                    self.root.update()
                except tk.TclError:
                    if self.root is not None:
                        self.root = None
            return board_img
        else:
            print("Game Over. Score: " + str(self.E.line_count))
        return None

    def update_board(self):
        win_state = self.E.step(self.command)
        self.raw_state = self.E.board_state
        if win_state == 0:
            cur_frame = self.display_frame()
            return 0, cur_frame
        else:
            cur_frame = None
            if self.root is not None:
                self.root.destroy()
            return self.E.line_count, cur_frame

    def game_loop(self):
        state = 0
        while state == 0:
            time.sleep(.05)
            state, cur_frame = self.update_board()
        return state
#
# test = Interface()
# test.game_loop()




