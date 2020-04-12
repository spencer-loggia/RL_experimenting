import tkinter as tk
import eviroment
import numpy as np
from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image, ImageTk
import scipy.ndimage
import time

class Interface:
    def __init__(self, human_disp=True):
        self.E = eviroment.Env()
        self.raw_state = self.E.board_state
        if human_disp:
            self.root = Tk()
            np_img = np.array(self.raw_state) * int(255 / 2)
            np_img = scipy.ndimage.zoom(np_img, 8, order=0)
            img = ImageTk.PhotoImage(image=Image.fromarray(np_img), master=self.root)
            self.canvas = Canvas(self.root, width=np_img.shape[1], height=np_img.shape[0])
            self.canvas.create_image(1, 2, anchor="nw", image=img)
            self.canvas.pack()
            self.canvas.update_idletasks()
            self.canvas.update()
            self.command = ''
            self.frame = Frame(self.root, width=100, height=100)
            self.root.bind('<Left>', self.leftKey)
            self.root.bind('<Right>', self.rightKey)
            self.frame.pack()

    def leftKey(self, event):
        self.E.move_left()

    def rightKey(self, event):
        self.E.move_right()

    def display_frame(self):
        self.canvas.delete('all')
        np_img = np.array(self.raw_state) * int(255 / 2)
        np_img = scipy.ndimage.zoom(np_img, 8, order=0)
        img = ImageTk.PhotoImage(image=Image.fromarray(np_img), master=self.root)
        self.canvas.create_image(1, 2, anchor="nw", image=img)
        self.canvas.pack()
        self.root.update_idletasks()
        self.root.update()

    def update_board(self):
        win_state = self.E.step(self.command)
        self.raw_state = self.E.board_state
        self.display_frame()
        self.command = ''

    def game_loop(self):
        while True:
            time.sleep(.2)
            self.update_board()



GUI = Interface()
GUI.game_loop()



