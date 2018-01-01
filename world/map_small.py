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

inp_dim=711
out_dim = 66
hidden = int(sys.argv[1])
exp_name = sys.argv[2]

arch = str(hidden) + '_6layerReLu'
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
    A = np.loadtxt('/home/sirisha.rallabandi/data/tts_stuff/input_full/' + train_file + '.lab')
    i_l = len(A)
    B = np.loadtxt('/home/sirisha.rallabandi/data/tts_stuff/output_full/' + train_file + '.ccoeffs')
    o_l = len(B)
    if i_l == o_l:
      for (a,b) in zip(A,B):
        train_input.append(a)
        train_output.append(b)
    else:
      print "Discarded ", train_file

for valid_file in files_test:
    A = np.loadtxt('/home/sirisha.rallabandi/data/tts_stuff/input_full/' + valid_file + '.lab')
    i_l = len(A)
    B = np.loadtxt('/home/sirisha.rallabandi/data/tts_stuff/output_full/' + valid_file + '.ccoeffs')
    o_l = len(B)
    if i_l == o_l:
      valid_input.append(A)
      valid_output.append(B)
      valid_files.append(valid_file)
    else:
      print "Discarded ", valid_file


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
        pass
        # If  first epoch, remove the log file
        if epoch == 0:
            g = open(logfile_name,'w')
            g.close()

        # Log the progress
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
        with open(logfile_name,'a') as g:
            g.write(msg + '\n')
        #test_model(self.model,input_scaler,output_scaler, epoch)
        
        #Save the model every 5 epochs
        if epoch % 15 == 1 and save_model:
             print self.model
             self.model.save(model_dir + '_' + arch + '.h5')


def test_model():
   # Test each file
   for (inp, out, fname) in valid_data:
       #inp = input_scaler.transform(inp)
       pred = model.predict(inp)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(validation_dir + '/' + fname + '.ccoeffs', pred)   
       np.savetxt(resynth_dir + '/' + fname + '.ccoeffs', out)
   for (inp, out, fname) in test_data:
       #inp = input_scaler.transform(inp)
       pred = model.predict(inp)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(test_dir + '/' + fname + '.ccoeffs', pred)  
       np.savetxt(resynth_dir + '/' + fname + '.ccoeffs', out)
   
input_scaler = preprocessing.StandardScaler().fit(train_input)
output_scaler = preprocessing.StandardScaler().fit(train_output)
#train_input = input_scaler.transform(train_input)
train_output = output_scaler.transform(train_output)

def train_model():

   global model
   # Create the model	
   model = Sequential()

   # INPUT LAYER
   model.add(Dropout(0.0, input_shape=(inp_dim,)))
   model.add(Dense(inp_dim,activation='relu'))

   # HIDDEN 1
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   # HIDDEN 2
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   # HIDDEN 3
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   # HIDDEN 4
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   # HIDDEN 5
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   # HIDDEN 6
   model.add(Dense(hidden,  activation='relu'))
   model.add(Dropout(0.2))

   model.add(Dense(out_dim, activation='relu'))

   # Compile the model
   sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
   model.compile(optimizer=sgd, loss='mse')
   model.summary()
   model.fit(train_input,train_output,epochs=40, batch_size=256, shuffle=True,callbacks=[LoggingCallback(logging.info)])

train_model()
test_model()


# Resynthesize
cmd = "./do_synth synth_world " + resynth_dir  + ' files.test'
os.system(cmd)

cmd = "./do_synth synth_world " + test_dir  + ' files.test'
os.system(cmd)

cmd = "./do_synth synth_world " + validation_dir  + ' files.test'
os.system(cmd)
