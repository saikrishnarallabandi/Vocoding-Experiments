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


print_flag = 0

class SequenceWise(torch.nn.Module):

   def __init__(self, module):
       super(SequenceWise, self).__init__()
       self.module = module

   def forward(self,x):
       t,n = x.size(0), x.size(1)
       x = x.contiguous().view(t * n, -1)
       x = self.module(x)
       x = x.view(t, n, -1)
       return x


class Model_LstmFc(torch.nn.Module):

    def __init__(self):
        super(Model_LstmFc, self).__init__()
        self.rnn = nn.LSTM(66, 256, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc1 = nn.Linear(512, 128) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = nn.Linear(128, 16)

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


class Model_LstmFc_v2(torch.nn.Module):

    def __init__(self):
        super(Model_LstmFc_v2, self).__init__()
        self.rnn = nn.LSTM(66, 256, num_layers=3, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc1 = SequenceWise(nn.Linear(512, 256)) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = SequenceWise(nn.Linear(256, 16))

    def forward(self,x):
        if print_flag:
          print("Shape input to RNN is ", x.shape) # [B,10,66]
        out, (h, cell) = self.rnn(x) # (batch_size, seq, input_size)
        out_fc1 = F.relu(self.fc1(out))
        out_fc2 = self.fc2(out_fc1)
        if print_flag :
          print("Shape output from  RNN is ", out.shape) # [B, 10, 512]
          print("Shape of output after fc1 : ", out_fc1.shape)          
          print("Shape of output after fc2 : ", out_fc2.shape)
        out_fc2_hat = out_fc2.view(out_fc2.size(0), out_fc2.size(1)*out_fc2.size(2))
        return out_fc2_hat
        sys.exit()

