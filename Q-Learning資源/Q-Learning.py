import os
import numpy as np
import pandas as pd
import time
from maze_env import Maze
from rl_brain import QLearningTable

def update():
    for episode in range(1000000000):
        observation = env.reset() #初始化
        count_action_time = 0

        while True:
            env.render()  #更新可視化環境

            action = RL.choose_action(str(observation), episode)  #根據state挑選action
            count_action_time += 1

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                print(f'第{episode}場遊戲, 走了{count_action_time}步')
                print(RL.q_table,'\n')
                break

    print('GAME OVER')
    env.destroy()

if __name__ == '__main__':

    # 切換工作路徑
    try:
        os.chdir('./Q-Learning資源')
        print(f'工作路徑切換為{os.getcwd()}')
    except OSError:
        print(f'工作路徑切換失敗，當前工作路徑為{os.getcwd()}')

    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))  #預設action_space = ['u', 'd', 'r', 'l']

    env.after(100, update)
    # env.after(1, update)
    env.mainloop()