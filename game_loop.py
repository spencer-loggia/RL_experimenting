import human_interface
import eviroment
import tkinter

class Game:
    def __init__(self, is_human=True):
        env = eviroment.Env()
        if is_human:
            self.hgui = human_interface.Container(env.board_state, env.width, env.height)

    def main_human_loop(self):
        win_state = 0
        while win_state == 0:

