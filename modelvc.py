import os, sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time

print_flag = 0

# Input Shape: (B, T_src, 66) Output Shape: (B, T_tgt, 66)
class Model_LstmFc_v3(torch.nn.Module):

    def __init__(self):
        super(Model_LstmFc_v3, self).__init__()
        self.rnn = nn.LSTM(256, 33, num_layers=1, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.cnn1 = nn.Conv1d(66,128,kernel_size = 5, stride=1,padding=2) # input channels, output channels
        self.cnn2 = nn.Conv1d(128,256,kernel_size = 5, stride=1,padding=2)

    def forward(self,x):

        x = x.double()
        if print_flag:
          print("Shape input to CNN is ", x.shape) # [B,T,66]

        # CNN expects (B,C,N)
        x.transpose_(1,2)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x.transpose_(1,2)
        if print_flag:
          print("Shape of output from CNN is ", x.shape) # (B,T,256)

        out, (h, cell) = self.rnn(x) # (batch_size, seq, input_size)
        if print_flag :
          print("Shape output from  RNN is ", out.shape) # [B, T, 66]
        return out

        out = out.contiguous().view(out.size(0), out.size(1)*16, out.size(2)/16) # [B, 160, 64]

        out_fc1 = F.relu(self.fc1(out)) #[B, 160, 128]
        if print_flag:
          print("The Shape of out_fc1: ", out_fc1.shape)

        out_fc1 = self.fc1_bn(out_fc1) # https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216
        out_fc2 = self.fc2(out_fc1)
        out_fc2 = self.fc2_bn(out_fc2)
        return F.log_softmax(out_fc2,dim=-1)


