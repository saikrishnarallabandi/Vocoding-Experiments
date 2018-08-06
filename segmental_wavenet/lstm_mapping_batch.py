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
from logger import Logger
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from model import *

'''
Resources:
https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/09-RNN.ipynb
'''

ccoeffs_folder = 'data_1msec'
wav_folder = ccoeffs_folder
train_file = ccoeffs_folder + '/train.txt'
test_file = 'test.txt'
exp_dir = '../exp/A'
if not os.path.exists(exp_dir):
   os.makedirs(exp_dir)

print_flag = 0
num_train = 10000
batch_size = 1
epochs = 100
saveplot_flag = 1
log_flag = 1

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

           if ctr % 1000 == 1:
               print("Read ", len(self.ccoeffs_array), " sentences")


      def __getitem__(self, index):
           return self.ccoeffs_array[index], self.wav_array[index]


      def __len__(self):
           return len(self.ccoeffs_array) 

def test(epoch):
  test_array = []
  for (a,b) in test_loader:
   
     if torch.cuda.is_available():
         A = Variable(a).cuda()
         B = Variable(b.unsqueeze_(0)).cuda()
     else:
         A = Variable(a)
         B = Variable(b.unsqueeze_(0))

     B_out = net(A).squeeze(0).cpu().detach().numpy()
     for w in B_out.T:
       test_array.append(w)
     w = np.array(test_array) * 1.0
     sf.write(exp_dir + '/test_epoch' + str(epoch).zfill(3) + '.wav', np.asarray(w), 16000,format='wav',subtype="PCM_16")    
     if epoch % 10 == 1:
        saveplot_flag = 1
     else:
        saveplot_flag = 0

     if saveplot_flag :
          axes = plt.gca()
          axes.set_ylim([-0.9,0.9])
          plt.plot(w)
          plt.savefig(exp_dir + '/' + str(epoch).zfill(3) + '-plot_noquantization.png')
          plt.close()

wks_train = arctic_dataset(ccoeffs_folder, wav_folder, train_file)
train_loader = DataLoader(wks_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
wks_test = arctic_dataset(ccoeffs_folder, wav_folder, test_file)
test_loader = DataLoader(wks_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4
                         )


net = Model_LstmFc_v2()
net.double()
if torch.cuda.is_available():
   net.cuda()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
logger = Logger('./logs') 

time_now = time.time()

def main():
 updates = 0
 for epoch in range(epochs):
  epoch_loss = 0
  for i, (a,b) in enumerate(train_loader):
     updates += 1
     if print_flag:
        print ("The shape of input is ", a.shape, b.shape, " done?")
     if torch.cuda.is_available():
        A = Variable(a).cuda()
        B = Variable(b.unsqueeze_(-1)).cuda()
     else:
        A = Variable(a)
        B = Variable(b.unsqueeze_(-1))
     B_out = net(A)
     B_out.unsqueeze_(-1)
     if print_flag:
        print("The shape of output is ", B_out.shape)
     loss = mse_loss(B_out, B)
     epoch_loss += loss.cpu().data[0].numpy()
     optimizer.zero_grad()
     loss.backward()       
     optimizer.step()
     if log_flag:  
        logger.scalar_summary('TrainLoss', epoch_loss * 1.0 / ((i+1)* batch_size) , updates)
     if i % 1000 == 1:
         print ("   Loss after batch ", i, " : ", epoch_loss/ ((i+1)* batch_size) * 1.0)
  print ("Loss after epoch ", epoch, " : ", epoch_loss * 1.0 / ((i+1)*batch_size))  
  time_current = time.time()
  print("    Average Time taken per epoch: ", (time_current - time_now) * 1.0 / (1+epoch))
  test(epoch)         
  print('\n')

main()
