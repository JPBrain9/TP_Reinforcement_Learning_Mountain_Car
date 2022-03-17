# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:58:28 2022

@author: Julien Panteri
"""
# off-policy
from RL import RL
from config import E_GREEDY

class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, E_GREEDY)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update