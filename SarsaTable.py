# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:05:23 2022

@author: Julien Panteri
"""
# on-policy
from RL import RL
from config import E_GREEDY
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, E_GREEDY)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update