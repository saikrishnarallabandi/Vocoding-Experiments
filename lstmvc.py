import numpy as np
import sys, os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *

# Locations
src_folder = '../feats/VCC2SF1'
tgt_folder = '../feats/VCC2TF1'

src_files = sorted(os.listdir(src_folder))
tgt_files = sorted(os.listdir(tgt_folder))

# 
batch_size = 1

class vcc_dataset(Dataset):

      def __init__(self, src_array, tgt_array):
          self.src_array = src_array
          self.tgt_array = tgt_array

      def __getitem__(self, index):
           x = np.loadtxt(src_folder + '/' + self.src_array[index])
           y = np.loadtxt(tgt_folder + '/' + self.tgt_array[index])
           x_len = x.shape[0]
           y_len = y.shape[0]
           if x_len < y_len:
               return x, y[:x_len]
           elif y_len < x_len:
               return x[:y_len], y
           else:
               print (" This cannot  happen you fool!!")
           return np.loadtxt(src_folder + '/' + self.src_array[index]),  np.loadtxt(tgt_folder + '/' + self.tgt_array[index])

      def __len__(self):
           return len(self.src_array)
       


vcc_train = vcc_dataset(src_files, tgt_files)
train_loader = DataLoader(vcc_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0
                         )
net = Model_LstmFc_v3()
net.double()
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum= 0.9)
mse_loss = nn.MSELoss()

total_loss = 0
for i,(a,b) in enumerate(train_loader):
    a,b = Variable(a), Variable(b)
    #print("Shapes of a and b: ", a.shape, b.shape)
    c = net(a.double())
    loss = mse_loss(c, b)   
    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()       
    optimizer.step()
   
    if i % 10 == 1:
         print ("   Loss after batch ", i, " : ", total_loss* 1.0 / (i+1))
