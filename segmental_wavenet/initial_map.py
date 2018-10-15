import os
import numpy as np
from keras.utils import to_categorical
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable

'''

Resources: 
https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/09-RNN.ipynb

'''

ccoeffs_folder = 'data_new'
wav_folder = 'data_new'
train_file = 'data_new/train.txt'

print_flag = 0
num_train = 10000

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Conv1d(66,1,kernel_size=5,stride=1,padding=2) # Input channels, Output channels, kernel size, stride, padding        
        self.rnn = nn.LSTM(10, 160, batch_first=True) # input_dim, hidden_dim
        self.fc = nn.Linear(128, 66) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch

    def forward(self,x):
        # CNN expects (B,C,N)
        x.transpose_(1,2)
        x = self.cnn(x)
        if print_flag:
          print("Shape after CNN is ", x.shape) # [1, 1, 10])
          print("Shape input to RNN is ", x.shape) 
        x, h = self.rnn(x) # (batch_size, seq, input_size)
        if print_flag :    
          print("Shape output from  RNN is ", x.shape) # [1, 52, 1056]
        return x


class arctic_dataset(Dataset):
      def __init__(self, ccoeffs_folder, wav_folder, train_file):
        self.train_file = train_file
        self.ccoeffs_folder = ccoeffs_folder
        self.wav_folder = wav_folder
        self.ccoeffs_array = []
        self.wav_array= []
        f = open(self.train_file)
        ctr  = 0
        for line in f: # arctic_a0001-audio-0000.npy|arctic_a0001-mel-0000.npy|160|N/A|0
         if ctr < num_train:
           ctr += 1
           line = line.split('\n')[0].split('|')
           wav_fname = line[0]
           ccoeffs_fname = line[1]

           ccoeffs = np.load(ccoeffs_folder + '/' + ccoeffs_fname)
           self.ccoeffs_array.append(ccoeffs)

           wav = np.load(wav_folder + '/' + wav_fname)
           self.wav_array.append(wav)

           print(len(self.ccoeffs_array))

      def __getitem__(self, index):
           return self.ccoeffs_array[index], self.wav_array[index]


      def __len__(self):
           return len(self.ccoeffs_array) 



wks_train = arctic_dataset(ccoeffs_folder, wav_folder, train_file)
train_loader = DataLoader(wks_train,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                         )
net = Net()
net.double()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

for epoch in range(10):
  epoch_loss = 0
  for (a,b) in train_loader:
     if print_flag:
        print ("The shape of input is ", a.shape, b.shape)
     A = Variable(a)
     B = Variable(b.unsqueeze_(0))
     B_out = net(A)
     if print_flag:
        print("The shape of output is ", B_out.shape)
     loss = mse_loss(B_out, B)
     epoch_loss += loss.data[0].numpy()
     optimizer.zero_grad()
     loss.backward()       
     optimizer.step()  
  print ("Loss after epoch ", epoch, " : ", epoch_loss)  
         
