import os, sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import soundfile as sf
from logger import Logger
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time



class LstmFc(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(66, 256, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc1 = nn.Linear(512, 256) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = nn.Linear(256, 160)

    def forward(self,x):
        if print_flag:
          print("Shape input to RNN is ", x.shape) # [1,10,66]
        out, (h, cell) = self.rnn(x) # (batch_size, seq, input_size)
        h_hat = h.view( h.shape[1], h.shape[0]*h.shape[2])
        if print_flag :
          print("Shape output from  RNN is ", x.shape) # [1, 10, 256]
          print("Shape of hidden from RNN is ", h.shape)
          print("Shape of RNN cell: ", cell.shape)
          print("Modified shape of RNN hidden: ", h_hat.shape)
        return self.fc2(F.relu(self.fc1(h_hat))) #.tanh()


