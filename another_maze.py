import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 30
MAZE_H = 12
MAZE_W = 12

class Mz(tk.Tk, object):

    def __init__(self):
        super(Mz, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 9
        self.escapeOrigin = np.array([6, 6])
        self.angentOriginList = [np.array([5, 8]), np.array([8, 4]), np.array([5, 3])]
        self.angentList = []
        self.title('Pursuit-Escape')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, (MAZE_H+3) * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # draw grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for angentOrigin in self.angentOriginList:
            self.angentList.append(self.canvas.create_rectangle(
                angentOrigin[0]*UNIT - 10, angentOrigin[1]*UNIT - 10,
                angentOrigin[0]*UNIT + 10, angentOrigin[1]*UNIT + 10,
                fill='red'
            ))

        self.escape = self.canvas.create_oval(
            self.escapeOrigin[0]*UNIT-10, self.escapeOrigin[1]*UNIT-10,
            self.escapeOrigin[0]*UNIT+10, self.escapeOrigin[1]*UNIT+10,
            fill='green'
        )

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(1)
        for angent in self.angentList:
            self.canvas.delete(angent)

        self.angentList.clear()

        for angentOrigin in self.angentOriginList:
            self.angentList.append(self.canvas.create_rectangle(
                angentOrigin[0]*UNIT - 10, angentOrigin[1]*UNIT - 10,
                angentOrigin[0]*UNIT + 10, angentOrigin[1]*UNIT + 10,
                fill='red'
            ))

        self.canvas.delete(self.escape)
        self.escape = self.canvas.create_oval(
            self.escapeOrigin[0]*UNIT-10, self.escapeOrigin[1]*UNIT-10,
            self.escapeOrigin[0]*UNIT+10, self.escapeOrigin[1]*UNIT+10,
            fill='green'
        )

        observation = np.array([])
        observation = np.hstack((observation, (np.array(self.canvas.coords(self.angentList[0])))[:2]))
        for angent in self.angentList:
            observation = np.hstack((observation, np.array(self.canvas.coords(angent))[:2]-np.array(self.canvas.coords(self.escape))[:2]))


        observation /= UNIT
        observation = np.hstack((0, observation))
        return observation

    def step(self, num, action):
        s = self.canvas.coords(self.angentList[num])
        e = self.canvas.coords(self.escape)
        distance = np.sqrt(((np.array(s) - np.array(e))**2).sum()) / UNIT
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= 2*UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += 2*UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += 2*UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= 2*UNIT

        self.canvas.move(self.angentList[num], base_action[0], base_action[1])
        next_coords = self.canvas.coords(self.angentList[num])
        _distance = np.sqrt(((np.array(next_coords) - np.array(e))**2).sum()) /UNIT

        if next_coords == e:
            reward = 2
            done = True
        elif _distance < distance:
            reward = 1
            done = False
        elif _distance >= distance:
            reward = -1
            done = False
        else:
            reward = 0
            done = False

        _observation = np.array([])
        _observation = np.hstack((_observation, (np.array(s))[:2]))
        for angent in self.angentList:
            _observation = np.hstack(
                (_observation, np.array(self.canvas.coords(angent))[:2] - np.array(self.canvas.coords(self.escape))[:2]))
        _observation /= UNIT

        if num == 2:
            _observation = np.hstack((0, _observation))
        else:
            _observation = np.hstack((num+1,_observation))

        if (action == 0) & (s[1] > UNIT):
            reward = -1
        elif (action == 1) & (s[1] < (MAZE_H - 1) * UNIT):
            reward = -1
        elif (action == 2) & (s[0] < (MAZE_W - 1) * UNIT):
            reward = -1
        elif (action == 3) & (s[0] > UNIT):
            reward = -1

        return _observation, reward, done


    def doEscape(self):
        n = int(np.random.random()*4)
        e = self.canvas.coords(self.escape)
        base_action = np.array([0, 0])
        if n == 0:   # up
            if e[1] > UNIT:
                base_action[1] -= 2*UNIT
        elif n == 1:   # down
            if e[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += 2*UNIT
        elif n == 2:   # right
            if e[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += 2*UNIT
        elif n == 3:   # left
            if e[0] > UNIT:
                base_action[0] -= 2*UNIT
        self.canvas.move(self.escape, base_action[0], base_action[1])

    def render(self):
        time.sleep(0.01)
        self.update()
