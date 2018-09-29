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
#from logger import Logger
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from model import *
from torch.legacy.nn import ClassNLLCriterion
'''
Resources:
https://github.com/MagaliDrumare/How-to-learn-PyTorch-NN-CNN-RNN-LSTM/blob/master/09-RNN.ipynb
'''

ccoeffs_folder = 'data_new'
wav_folder = 'data_new'
train_file = 'train.txt'
test_file = 'test.txt'

print_flag = 0
num_train = 100000
batch_size = 8
epochs = 100
saveplot_flag = 0
log_flag = 0


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

     if ctr % 10000 == 1:
         print (" Loaded ", ctr, " files")


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
     b = b.view(-1)
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     B_out = net(A).squeeze(0).cpu().detach().numpy()
     B_out = np.argmax(B_out,-1)
     num = np.random.randint(0,150)
     if i% 100 == 1:
         print("Test sample ", i, " : ", num, b.numpy()[num:num+10], B_out[num:num+10])

     if print_flag:
       print ("Shape of B_out in test: ", B_out.shape)
       #sys.exit()
     w = B_out.T
     #print(w)
     np.save("data_synthesis/arctic_a0001" + '-audio-' + str(i).zfill(4) + '.npy',w)     
  cmd = 'python2 test_synthesis_mulaw.py arctic_a0001 ' + str(epoch)
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


net = Model_LstmFc_v3()
net = Model_CnnFc()
net = Model_AllCnn()
net.double()
if torch.cuda.is_available():
   net.cuda()
#mse_loss = nn.MSELoss()
ce_loss = F.nll_loss
ce_loss = nn.NLLLoss()
#ce_loss = ClassNLLCriterion
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
time_now = time.time()
#logger = Logger('./logs')
updates = 0

def valid(epoch):
  net.eval()
  epoch_loss = 0
  for i, (a,b) in enumerate(valid_loader):
     A = Variable(a) #.cuda()
     B = Variable(b) #.unsqueeze_(-1)) #.cuda()
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     B_out = net(A)
     B_out = B_out.view(B_out.size(0)*B_out.size(1),B_out.size(2))
     B = B.view(B.size(0)*B.size(1), 1).view(-1)

     loss = ce_loss(B_out, B)

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
        print ("The shape of input is ", a.shape, " and ", b)
     A = Variable(a) #.cuda()
     B = Variable(b) #.unsqueeze_(-1)) #.cuda()
     if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
     #B = B.double()
     B_out = net(A)
     #B_out.unsqueeze_(-1)
     B_out = B_out.view(B_out.size(0)*B_out.size(1),B_out.size(2))
     B = B.view(B.size(0)*B.size(1), 1).view(-1)
     #B = B.contiguous().view(-1)
     if print_flag:
        print("The shape of output is ", B_out.shape)
        print("The shape of B is ", B.shape, B)
     loss = ce_loss(B_out, B) #.double())
     epoch_loss += loss.cpu().item()

     optimizer.zero_grad()
     loss.backward()       
     optimizer.step()

     if log_flag:  
        logger.scalar_summary('Train_Loss', epoch_loss* 1.0 / (i+1) , updates)
     if i % 10 == 1:
         #print ("   Loss after batch ", i, " : ", epoch_loss* 1.0 / (i+1), updates)
         #test(epoch)
         cnt += 1
         #if cnt == 2:
         #  break
         #test(epoch)
         #sys.exit()

  print ("Training Loss after epoch ", epoch, " : ", epoch_loss* 1.0 / (i+1))
  if log_flag:  
        logger.scalar_summary('Training Loss per Epoch', epoch_loss* 1.0 / (i+1) , epoch)
        #logger.scalar_summary('Learning Rate', epoch_loss* 1.0 / (i+1) , epoch)
  validation_loss = valid(epoch)
  scheduler.step(validation_loss)  
  time_current = time.time()
  print("Average Time taken per epoch: ", (time_current - time_now) * 1.0 / (1+epoch))
  test(epoch)         
  print('\n')

main()
