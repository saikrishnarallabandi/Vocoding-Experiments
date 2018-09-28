import os, sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import soundfile as sf
# from logger import Logger
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
        self.rnn = nn.LSTM(66, 256, num_layers=2, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc1 = SequenceWise(nn.Linear(512, 256)) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = SequenceWise(nn.Linear(256, 16))
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self,x):
        if print_flag:
          print("Shape input to RNN is ", x.shape) # [B,10,66]
        out, (h, cell) = self.rnn(x) # (batch_size, seq, input_size)
        out_fc1 = F.relu(self.fc1(out))
        out_fc1 = self.fc2_bn(out_fc1) # https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216
        out_fc2 = self.fc2(out_fc1)
        out_fc2 = self.fc2_bn(out_fc2)
        if print_flag :
          print("Shape output from  RNN is ", out.shape) # [B, 10, 512]
          print("Shape of output after fc1 : ", out_fc1.shape)          
          print("Shape of output after fc2 : ", out_fc2.shape)
        out_fc2_hat = out_fc2.view(out_fc2.size(0), out_fc2.size(1)*out_fc2.size(2))
        return out_fc2_hat
        #sys.exit()
         
# Input Shape: (B, 10, 66) Output Shape: (B, 160, 256)
class Model_CnnFc(torch.nn.Module):

    def __init__(self):
        super(Model_CnnFc, self).__init__()
        self.fc1 = SequenceWise(nn.Linear(64, 128)) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = SequenceWise(nn.Linear(128, 256))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.cnn1 = nn.Conv1d(66,128,kernel_size = 5, stride=1,padding=2) # input channels, output channels
        self.cnn2 = nn.Conv1d(128,256,kernel_size = 5, stride=1,padding=2)
        self.cnn3 = nn.Conv1d(10,160,kernel_size = 5, stride=1, padding=2)

    def forward(self,x):

        if print_flag:
          print("Shape input to CNN is ", x.shape) # [B,10,66]

        # CNN expects (B,C,N)
        x.transpose_(1,2)

        x = F.relu(self.cnn1(x))
        x = self.bn1(x)
        x = F.relu(self.cnn2(x))
        x = self.bn2(x)

        x.transpose_(1,2)
        x = self.cnn3(x)
        if print_flag:
          print("Shape of output from CNN is ", x.shape) # (B,10,256)

        return F.log_softmax(x,dim=-1)

         
# Input Shape: (B, 10, 66) Output Shape: (B, 160, 256)
class Model_AllCnn(torch.nn.Module):

    def __init__(self):
        super(Model_AllCnn, self).__init__()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # Dimension changing CNNs
        self.cnn1 = nn.Conv1d(66,128,kernel_size = 7, stride=1,padding=3) # input channels, output channels
        self.cnn2 = nn.Conv1d(128,256,kernel_size = 7, stride=1,padding=3)
 
        # Autoregressive CNNs
        self.cnn3 = nn.Conv1d(10,160,kernel_size = 5, stride=1, padding=2)

    def forward(self,x):

        if print_flag:
          print("Shape input to CNN is ", x.shape) # [B,10,66]

        # CNN expects (B,C,N)
        x.transpose_(1,2)
        x = F.relu(self.cnn1(x))
        x = self.bn1(x)
        x = F.relu(self.cnn2(x))
        x = self.bn2(x)
        x.transpose_(1,2)
        print("Shape of x after CNN dimensionality modification: ", x.shape)
        sys.exit()
        x = self.cnn3(x)
        if print_flag:
          print("Shape of output from CNN is ", x.shape) # (B,10,256)

        return F.log_softmax(x,dim=-1)

