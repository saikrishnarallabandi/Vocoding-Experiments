import os
import numpy as np
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.constraints import maxnorm
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.models import Model, Sequential
import numpy as np
from keras.models import load_model
import os,sys
from sklearn import preprocessing
import pickle, logging
from keras.callbacks import *
import random
import sys, os
sys.path.append('/home/srallaba/hacks/repos/clustergen_steroids')
from building_blocks.DNNs import *
import dynet as dy
import numpy as np

inp_dim=711
out_dim = 66
hidden = int(sys.argv[1])
exp_name = sys.argv[2]
dropout = float(sys.argv[3])

arch = str(hidden) + '_6layertanh_' + str(dropout) + 'dropout_sgd_dynet'
logfile_name = exp_name + '/logs/log_' + arch + '.log'
g = open(logfile_name,'w')
g.close()
model_dir = exp_name + '/models/'
test_dir = exp_name + '/test/' + arch
resynth_dir = exp_name + '/resynth/' + arch
validation_dir = exp_name + '/validation/' + arch
save_model = 1

for k in [model_dir, test_dir, resynth_dir, validation_dir]:
   if not os.path.exists(k):
      os.makedirs(k) 

# Declare train and test files

files_train = []
files_test = []
train_file = 'files.train'
test_file = 'files.test'

f = open(train_file)
for line in f:
   line = line.split('\n')[0]
   files_train.append(line)
f.close()

f = open(test_file)
for line in f:
   line = line.split('\n')[0]
   files_test.append(line)


# Load train and validation data
train_input = []
train_output = []
valid_input = []
valid_output = []
valid_files = []

for train_file in files_train:
    A = np.loadtxt('../data/input_full/' + train_file + '.lab')
    i_l = len(A)
    B = np.loadtxt('../data/output_full/' + train_file + '.ccoeffs')
    o_l = len(A)
    diff_l = o_l - i_l
    for i in range(diff_l):
       A.append(A[-1])
    for (a,b) in zip(A,B):
       train_input.append(a)
       train_output.append(b)

for valid_file in files_test:
    A = np.loadtxt('../data/input_full/' + valid_file + '.lab')
    i_l = len(A)
    B = np.loadtxt('../data/output_full/' + valid_file + '.ccoeffs')
    o_l = len(A)
    diff_l = o_l - i_l
    for i in range(diff_l):
       A.append(A[-1])
    valid_input.append(A)
    valid_output.append(B)
    valid_files.append(valid_file)

num_valid = len(valid_files)
valid_data = zip(valid_input, valid_output, valid_files)
random.shuffle(valid_data)
test_data = valid_data[0:int(num_valid/2)]
valid_data = valid_data[int(num_valid)/2+1:]

train_input = np.array(train_input)
train_output = np.array(train_output)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        # If  first epoch, remove the log file
        if epoch == 0:
            g = open(logfile_name,'w')
            g.close()

        # Log the progress
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
        with open(logfile_name,'a') as g:
            g.write(msg + '\n')
        #test_model()
        
        #Save the model every 5 epochs
        if epoch % 15 == 1 and save_model:
             print self.model
             self.model.save(model_dir + 'model_' + arch + '.h5')


def test_model():
   # Test each file
   for (inp, out, fname) in valid_data:
       #inp = input_scaler.transform(inp)
       pred = []
       for input_frame in inp:
           input_frame = dy.inputTensor(input_frame)
           pred_frame = dnn.predict(input_frame)
           pred.append(np.asarray(pred_frame.value()))
       pred = np.asarray(pred)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(validation_dir + '/' + fname + '.ccoeffs', pred)   
       np.savetxt(resynth_dir + '/' + fname + '.ccoeffs', out)
   for (inp, out, fname) in test_data:
       #inp = input_scaler.transform(inp)
       pred = []
       for input_frame in inp:
           input_frame = dy.inputTensor(input_frame)
           pred_frame = dnn.predict(input_frame)
           pred.append(np.asarray(pred_frame.value()))
       pred = np.asarray(pred)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(test_dir + '/' + fname + '.ccoeffs', pred)  
       np.savetxt(resynth_dir + '/' + fname + '.ccoeffs', out)
   

input_scaler = preprocessing.StandardScaler().fit(train_input)
output_scaler = preprocessing.StandardScaler().fit(train_output)
#train_input = input_scaler.transform(train_input)
train_output = output_scaler.transform(train_output)


# Hyperparameters 
units_input = inp_dim
units_hidden = int(hidden)
units_output = out_dim

m = dy.Model()
dnn = FeedForwardNeuralNet(m, [units_input, [units_hidden, units_hidden, units_hidden, units_hidden], units_output, [dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh]])
trainer = dy.SimpleSGDTrainer(m)
update_params = 32
num_epochs = 40


def train_model():

   train_data = zip(train_input, train_output)
   num_train = len(train_data)
   startTime = time.time()

   # Loop over the training instances and call the mlp
   for epoch in range(num_epochs):
     start_time = time.time()
     print " Epoch ", epoch
     train_loss = 0
     random.shuffle(train_data)
     K = 0
     count = 0
     frame_count = 0
     for (i,o) in train_data:
        if debug and count == 0:
          print "Input dimensions: ", len(i)
          print "Number of frames: ", num_train
        count = 1
        K += 1
        dy.renew_cg()
        frame_count += 1
        count += 1
        loss = dnn.calculate_loss(dy.inputTensor(i), dy.inputTensor(o))
        train_loss += loss.value()
        loss.backward()
        if debug and frame_count % int(0.1*num_train) == 1:
           print "   Train Loss after processing " +  str(frame_count) + " number of frames : " +  str(float(train_loss/frame_count))
        if frame_count % update_params == 1:
            trainer.update() 
     end_time = time.time()
     duration = end_time - start_time
     start_time = end_time
     with open(logfile_name, 'a') as g:
        g.write("Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)) + '\n')
     print "Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)), " with ", frame_count, " frames, in ", float((end_time - startTime)/60)  , " minutes "  
     print "I think I will run for another ", float( duration * ( num_epochs - epoch) / 60 ), " minutes "
     print '\n'

train_model()
test_model()


# Resynthesize
cmd = "./do_synth synth_world " + resynth_dir  + ' files.test'
os.system(cmd)

cmd = "./do_synth synth_world " + test_dir  + ' files.test'
os.system(cmd)

cmd = "./do_synth synth_world " + validation_dir  + ' files.test'
os.system(cmd)
