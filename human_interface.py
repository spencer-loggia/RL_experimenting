import tkinter
import numpy as np
from tkinter import Tk, Canvas, Frame, BOTH

class Container(Frame):
    def __init__(self, board_state, w, h):
        super().__init__()
        self.height = h
        self.width = w
        self.board_state = board_state
        self.initUI()

    def initUI(self):
        self.master.title("RUN")
        self.pack(fill=BOTH, expand=1)
        self.render_board()

    def render_board(self, canvas):
        canvas = Canvas(self)
        pix_w = 400 / self.width
        pix_h = 800 / self.height
        for i in range(self.height):
            for j in range(self.width):
                if self.board_state[i][j] == 1:
                    x = pix_w * j
                    y = pix_h * i
                    canvas.create_rectangle(x, y, x+pix_w, y+pix_h, outline="#fb0", fill="#fb0")
                    canvas.pack(fill=BOTH, expand=1)
                elif self.board_state[i][j] == 2:
                    x = pix_w * j
                    y = pix_h * i
                    canvas.create_rectangle(x, y, x + pix_w, y + pix_h, outline="#05f", fill="#05f")
                    canvas.pack(fill=BOTH, expand=1)



def main():
    root = Tk()
    ex = Container([[0,1],[0,0]], 2, 2)
    root.geometry("400x800+300+300")
    root.mainloop()

if __name__ == '__main__':
    main()