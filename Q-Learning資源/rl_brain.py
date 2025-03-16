import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate  #學習率
        self.gamma = reward_decay #久遠重要性，數字越高代表越久以前的行為參考程度越高
        self.epsilon = e_greedy  #往好的方向走的貪婪度

        '''
        為了調用訓練過的Q-table，判斷資料夾內有無
        Q-table資料檔「trained_q_table.csv」
        '''
        try:
            df = pd.read_csv(
                'trained_q_table.csv',
                skiprows=1, # 檔案表頭讀取後會變成文字造成loc比對不到對應欄位，故不使用
                names=[0, 1, 2, 3], # 手動設定表頭，讓欄位名稱為整數，以利loc比對
                index_col=0 # 第一欄作為index，格式為str為str
                )
            self.q_table = df
            print('成功匯入Q-Table:')
            print(self.q_table)
        except:
            self.q_table = pd.DataFrame(columns=self.actions)
            print('未匯入Q-Table，從頭開始訓練')

    def choose_action(self, observation, episode):
        self.check_state_exist(observation)

        # action selection
        if episode % 10000 == 0:
            if episode == 0:
                pass

            else:
                if self.epsilon < 0.9:
                    self.epsilon *= 1.01
                    print(f'第{episode}場遊戲，調整貪婪指數為{self.epsilon}')

                else:
                    self.epsilon = 0.9

        if np.random.uniform() < self.epsilon: #當貪婪發生
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 當值相同時用random重排前後，再用idxmax挑出最先出現的，已實現隨機性
            action = state_action.idxmax()  #選擇Q表中最大值
            
        else:   #貪婪未發生，隨機選擇行為
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        pass

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = pd.concat([
                self.q_table,
                pd.DataFrame(
                    [[0] * len(self.actions)],  # 建立新的 row，所有值都是 0
                    columns=self.q_table.columns,  # 確保欄位名稱一致
                    index=[state]  # 設定索引名稱為 state
                )]
            )

class QLearningTable(RL):
    def __init__(
            self,
            actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9
            ):
        super(QLearningTable,self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        # Q-table更新方式1
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        
        # Q-table更新方式2
        self.q_table.loc[s,a] = ((1 - self.lr) * self.q_table.loc[s,a]) + (self.lr * (q_target - q_predict))
