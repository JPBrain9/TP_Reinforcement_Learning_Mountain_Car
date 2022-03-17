# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:07:21 2022

@author: Julien Panteri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# DQN
class Net(nn.Module):
    def __init__(self, action_n, state_n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_n, 50).to(device)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, action_n).to(device)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value