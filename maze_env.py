
import sys
import time
import numpy as np
import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.goals = 0
        # self.check_points = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell 1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        # hell 2
        hell2_center = origin + np.array([UNIT * 4, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
             hell2_center[0] - 15, hell2_center[1] - 15,
             hell2_center[0] + 15, hell2_center[1] + 15,
             fill='black')

        # hell 3
        hell3_center = origin + np.array([UNIT, UNIT * 2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        
        # hell 4
        hell4_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        
        # hell 5
        hell5_center = origin + np.array([UNIT, UNIT])
        self.hell5 = self.canvas.create_rectangle(
             hell5_center[0] - 15, hell5_center[1] - 15,
             hell5_center[0] + 15, hell5_center[1] + 15,
             fill='black')

        # hell 6
        hell6_center = origin + np.array([UNIT * 0, UNIT * 5])
        self.hell6 = self.canvas.create_rectangle(
             hell6_center[0] - 15, hell6_center[1] - 15,
             hell6_center[0] + 15, hell6_center[1] + 15,
             fill='black')

        # hell 7
        hell7_center = origin + np.array([UNIT * 3, UNIT * 5])
        self.hell7 = self.canvas.create_rectangle(
             hell7_center[0] - 15, hell7_center[1] - 15,
             hell7_center[0] + 15, hell7_center[1] + 15,
             fill='black')

        # hell 8
        hell8_center = origin + np.array([UNIT * 4, UNIT * 0])
        self.hell8 = self.canvas.create_rectangle(
             hell8_center[0] - 15, hell8_center[1] - 15,
             hell8_center[0] + 15, hell8_center[1] + 15,
             fill='black')

        # hell 9
        hell9_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell9 = self.canvas.create_rectangle(
             hell9_center[0] - 15, hell9_center[1] - 15,
             hell9_center[0] + 15, hell9_center[1] + 15,
             fill='black')

        # hell 10
        hell10_center = origin + np.array([UNIT * 5, UNIT * 4])
        self.hell10 = self.canvas.create_rectangle(
             hell10_center[0] - 15, hell10_center[1] - 15,
             hell10_center[0] + 15, hell10_center[1] + 15,
             fill='black')
                
        # create tmp_reward1
        # tmp_reward1_center = origin + np.array([UNIT * 3, UNIT * 1])
        # self.tmp_reward1 = self.canvas.create_oval(
        #     tmp_reward1_center[0] - 15, tmp_reward1_center[1] - 15,
        #     tmp_reward1_center[0] + 15, tmp_reward1_center[1] + 15,
        #     fill='green')
        
        # create tmp_reward2
        # tmp_reward2_center = origin + np.array([UNIT * 1, UNIT * 3])
        # self.tmp_reward2 = self.canvas.create_oval(
        #     tmp_reward2_center[0] - 15, tmp_reward2_center[1] - 15,
        #     tmp_reward2_center[0] + 15, tmp_reward2_center[1] + 15,
        #     fill='green')
        
        # create oval
        oval_center = origin + UNIT * 5
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            self.goals += 1
            print(f'走到終點{self.goals}次')
            # print(f'走到中繼點{self.check_points}次，走到終點{self.goals}次')

        # elif s_ in [self.canvas.coords(self.tmp_reward1), self.canvas.coords(self.tmp_reward2)]:
        #     reward = 0.5
        #     done = False
        #     self.check_points += 1

        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3), self.canvas.coords(self.hell4), self.canvas.coords(self.hell5),
                    self.canvas.coords(self.hell6), self.canvas.coords(self.hell7), self.canvas.coords(self.hell8), self.canvas.coords(self.hell9), self.canvas.coords(self.hell10)]:
            reward = -1
            done = True
            print(f'走到終點{self.goals}次')
            # print(f'走到中繼點{self.check_points}次，走到終點{self.goals}次')
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()