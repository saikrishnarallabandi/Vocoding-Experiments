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

phones_file = '../voices/cmu_us_awb_arctic/ehmm/etc/txt.phseq.data'
ccoeffs_folder = '../feats/awb_5msec'
ccoeffs_files = sorted(os.listdir(ccoeffs_folder))
p = defaultdict(int)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Conv1d(43,66,kernel_size=5,stride=1,padding=2) # Input channels, Output channels, kernel size, stride, padding        
        self.rnn = nn.LSTM(66, 1056, batch_first=True) # input_dim, hidden_dim
        self.fc = nn.Linear(128, 66) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch

    def forward(self,x):
        # CNN expects (B,C,N)
        x.transpose_(1,2)
        x = self.cnn(x)
        print("Shape after CNN is ", x.shape)
        x.transpose_(1,2)
        print("Shape input to RNN is ", x.shape)
        x, h = self.rnn(x) # (batch_size, seq, input_size)
        #x.transpose_(1,2)
        print("Shape output from  RNN is ", x.shape) # [1, 52, 1056]
        for i in range(4):
          batch_size = x.shape[0]
          channels = x.shape[1]
          seq_len = x.shape[2]
          x = x.contiguous().view(batch_size,channels*2, int(seq_len/2))
          print("Shape after ", i, " reshaping is ", x.shape)
        return x


class arctic_dataset(Dataset):
      
      def __init__(self, ccoeffs_folder, phones_file):
        self.phones_file = phones_file
        self.ccoeffs_folder = ccoeffs_folder
        self.phones_array = []
        self.ccoeffs_array = []
        self.filenames_array = []
        f = open(self.phones_file)
        phones_dict = {}
        ctr  = 0
        for line in f:
         if ctr < 10:
           ctr += 1
           line = line.split('\n')[0].split()
           fname = line[0]
           phones = line[1:]
           phones_numeric = []
           for phone in phones:
              phones_numeric.append(p[phone] +1)
           phones_onehot = to_categorical(phones_numeric, 43)
           ccoeffs = np.loadtxt(ccoeffs_folder + '/' + fname + '.ccoeffs_ascii')
           self.phones_array.append(phones_onehot)
           self.filenames_array.append(fname)
           self.ccoeffs_array.append(ccoeffs)
           print(len(self.ccoeffs_array))

      def __getitem__(self, index):
           return self.filenames_array[index], self.phones_array[index], self.ccoeffs_array[index]


      def __len__(self):
           return len(self.ccoeffs_array) 



wks_train = arctic_dataset(ccoeffs_folder, phones_file)
train_loader = DataLoader(wks_train,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                         )
net = Net()
net.double()

for (a,b,c) in train_loader:
   if c.shape[1] > b.shape[1] * 16:
     print ("The shape of input is ", a, b.shape, c.shape)
     B = Variable(b)
     C = Variable(c)
     c_out = net(b).detach().numpy()
     print("The shape of output is ", c_out.shape)
     c_hat = np.zeros(c.shape)
     print ("Shape of c_out[i] is ", c_out[0].shape, " and the shape of c_hat[i]: ", c_hat[0].shape)
     t = c_hat
     l = c_out[0].shape[0]
     m = c_hat[0].shape[0]
     print(l,m)
     if l > m: # The Neural net predicted more time steps. Cut them
           kk = np.zeros((c_out[0].shape[0] - c_hat[0].shape[0], 66))
           print("Chopping")           
     elif m > l: # The Neural net predicted less time steps. Append zeros.
           c_hat = np.zeros(c.shape)
           kk = np.zeros((c_hat[0].shape[0] - c_out[0].shape[0], 66))
           k = np.concatenate((c_out[0], kk),axis=0)
           print("Shape of k: ", k.shape)
           print("Appending ")
           c_hat[0] = k
     print("The shapes are ", c.shape, c_hat.shape)
     print("\n")

