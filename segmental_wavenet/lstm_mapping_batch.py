import os, sys
import numpy as np
from keras.utils import to_categorical
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import soundfile as sf
import matplotlib
from logger import Logger
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from model import *

'''
Resources:
https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/09-RNN.ipynb
'''

ccoeffs_folder = 'data_new'
wav_folder = 'data_new'
train_file = 'train.txt'
test_file = 'test.txt'

print_flag = 1
num_train = 400000
batch_size = 16
epochs = 100
saveplot_flag = 0
log_flag = 1

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(66, 256, batch_first=True, bidirectional=True) # input_dim, hidden_dim
        self.fc = nn.Linear(512, 160) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch

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
          print('\n')
        return self.fc(h_hat) #.tanh()


train_file = train_file
ccoeffs_folder = ccoeffs_folder
wav_folder = wav_folder
ccoeffs_array = []
wav_array= []
f = open(train_file)
ctr  = 0
for line in f: # arctic_a0001-audio-0000.npy|arctic_a0001-mel-0000.npy|160|N/A|0
  if ctr < num_train:
     ctr += 1
     line = line.split('\n')[0].split('|')
     wav_fname = line[0]
     ccoeffs_fname = line[1]

     ccoeffs = np.load(ccoeffs_folder + '/' + ccoeffs_fname)
     ccoeffs_array.append(ccoeffs)

     wav = np.load(wav_folder + '/' + wav_fname)
     wav_array.append(wav)

f = open(test_file)
ccoeffs_test = []
wav_test = []
for line in f:
     line = line.split('\n')[0].split('|')
     wav_fname = line[0]
     ccoeffs_fname = line[1]

     ccoeffs = np.load(ccoeffs_folder + '/' + ccoeffs_fname)
     ccoeffs_test.append(ccoeffs)

     wav = np.load(wav_folder + '/' + wav_fname)
     wav_test.append(wav)

from sklearn.utils import shuffle
ccoeffs_array, wav_array = shuffle(ccoeffs_array, wav_array)
from sklearn.model_selection import train_test_split
ccoeffs_train, ccoeffs_valid, wav_train, wav_valid = train_test_split(ccoeffs_array, wav_array,test_size=0.2)



class arctic_dataset(Dataset):
      def __init__(self, ccoeffs_array, wav_array):
          self.ccoeffs_array = ccoeffs_array
          self.wav_array = wav_array

      def __getitem__(self, index):
           return self.ccoeffs_array[index], self.wav_array[index]

      def __len__(self):
           return len(self.ccoeffs_array) 

def test(epoch):
  test_array = []
  for i, (a,b) in enumerate(test_loader):
     A = Variable(a) #.cuda()
     B = Variable(b.unsqueeze_(0)) #.cuda()
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     B_out = net(A).squeeze(0).cpu().detach().numpy()

     w = B_out.T
     np.save("data_synthesis/arctic_a0001" + '-audio-' + str(i).zfill(4) + '.npy',w)     
  cmd = 'python2 test_synthesis.py arctic_a0001 ' + str(epoch)
  os.system(cmd)




wks_train = arctic_dataset(ccoeffs_train, wav_train)
train_loader = DataLoader(wks_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
wks_valid = arctic_dataset(ccoeffs_valid, wav_valid)
valid_loader = DataLoader(wks_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
wks_test = arctic_dataset(ccoeffs_test, wav_test)
test_loader = DataLoader(wks_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4
                         )


#net = Net()
net = Model_LstmFc_v2()
net.double()
if torch.cuda.is_available():
   net.cuda()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
time_now = time.time()
logger = Logger('./logs')
updates = 0

def valid(epoch):
  net.eval()
  epoch_loss = 0
  for i, (a,b) in enumerate(valid_loader):
     A = Variable(a) #.cuda()
     B = Variable(b.unsqueeze_(-1)) #.cuda()
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     B_out = net(A)
     B_out.unsqueeze_(-1)
     loss = mse_loss(B_out, B)

     epoch_loss += loss.cpu().item()

     if log_flag:  
        logger.scalar_summary('Validation_Loss', epoch_loss* 1.0 / (i+1) , updates)
  print ("Validation Loss after epoch ", epoch, " : ", epoch_loss* 1.0 / (i+1))
  if log_flag:  
        logger.scalar_summary('Validation Loss per Epoch', epoch_loss* 1.0 / (i+1) , epoch) 
  return epoch_loss

def main():
 global updates
 for epoch in range(epochs):
  net.train()
  cnt = 0
  epoch_loss = 0
  for i, (a,b) in enumerate(train_loader):
     updates += 1
     if print_flag:
        print ("The shape of input is ", a.shape, b.shape, " done?")
     A = Variable(a) #.cuda()
     B = Variable(b.unsqueeze_(-1)) #.cuda()
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     B_out = net(A)
     B_out.unsqueeze_(-1)
     if print_flag:
        print("The shape of output is ", B_out.shape)
     loss = mse_loss(B_out, B)
     epoch_loss += loss.cpu().item()

     optimizer.zero_grad()
     loss.backward()       
     optimizer.step()

     if log_flag:  
        logger.scalar_summary('Train_Loss', epoch_loss* 1.0 / (i+1) , updates)
     if i % 60000 == 1:
         print ("   Loss after batch ", i, " : ", epoch_loss* 1.0 / (i+1), updates)
         #test(epoch)
         cnt += 1
         #if cnt == 2:
         #  break
         test(epoch)

  print ("Training Loss after epoch ", epoch, " : ", epoch_loss* 1.0 / (i+1))
  if log_flag:  
        logger.scalar_summary('Training Loss per Epoch', epoch_loss* 1.0 / (i+1) , epoch)
        #logger.scalar_summary('Learning Rate', epoch_loss* 1.0 / (i+1) , epoch)
  validation_loss = valid(epoch)
  scheduler.step(validation_loss)  
  time_current = time.time()
  print("Average Time taken per epoch: ", (time_current - time_now) * 1.0 / (1+epoch))
  #test(epoch)         
  print('\n')

main()
