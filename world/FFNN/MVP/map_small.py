import os, sys
import numpy as np
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.constraints import maxnorm
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.models import Model, Sequential
import numpy as np
import keras.backend as K
from keras.models import load_model
import os,sys
from sklearn import preprocessing
import pickle, logging
from keras.callbacks import *
from sklearn.metrics import mean_squared_error

'''

Assumes that input_full and output_full within a directory called data contain input and output features respectively. Loads some of them. 
How many are loaded can be tuned by changing the variable limit. Trains a neural net and generates predictions for each file.
Creates directories named test and resynth in the experiment folder ( declared in the first line in run.sh) and stores the predictions in them. 
If py_world is installed, also generates the wavefiles. Writes the log to 'log'
You can also make it synthesize wavefiles after every epoch of training.
Yeah for synthesis to work you need to have SPTK installed and stuff. 

Neural net trains for 4 iterations. You can save the model at any iteration for use later. Obviously this sounds crappy as we are using only 40 files and 4 iterations.
For comparision in terms of quality, listen to the wavefile in the resynth directory.

'''

exp_dir = sys.argv[1]

inp_dim=711
out_dim = 66
encoding_dim = 512


test_dir = exp_dir + '/test'
resynth_dir = exp_dir + '/resynth'

if not os.path.exists(test_dir):
   os.makedirs(test_dir)
   os.makedirs(resynth_dir)

input_files = [filename for filename in sorted(os.listdir('../data/input_full')) if filename.startswith("arctic")]
output_files = [filename for filename in sorted(os.listdir('../data/output_full')) if filename.startswith("arctic")]


train_input = []
train_output = []
valid_input = []
valid_output = []
valid_files = []
g = open('files.test','w')
lim = 50
# Load data
for i, (input_file, output_file) in enumerate(zip(input_files, output_files)):
  if i % 50 == 1:
    print "Processed ", i, " files"
  if i < lim:
    if '0009' in input_file:
      A = np.loadtxt('../data/input_full/' + input_file)
      i_l = len(A)
      B = np.loadtxt('../data/output_full/' + output_file) 
      o_l = len(B)
      if i_l == o_l:
         g.write(input_file.split('.')[0] + '\n')
         valid_input.append(A)
         valid_output.append(B)
         valid_files.append(output_file)
      else:
         print "Discarding ", input_file
    else:
      A = np.loadtxt('../data/input_full/' + input_file) 
      i_l = len(A) 
      B = np.loadtxt('../data/output_full/' + output_file) 
      o_l = len(B)
      
      if i_l == o_l:
         for (a,b) in zip(A,B):
            train_input.append(a)
            train_output.append(b)      
      else:
         print "Discarding ", input_file    

g.close()


def custom_mse(yTrue, yPred):
   print yPred
   return K.square(yTrue-yPred)

train_input = np.array(train_input)
train_output = np.array(train_output)
logfile_name=exp_dir + '/logs/log'

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
        #if epoch % 3 == 1 and save_model:
        #     print self.model
        #     self.model.save(exp_name + '/models/mvp.h5')


def test_model():
   # Test each file
   for (inp, out, fname) in zip(valid_input, valid_output, valid_files):
       inp = input_scaler.transform(inp)
       pred = model.predict(inp)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(test_dir + '/' + fname, pred)
       np.savetxt(resynth_dir + '/' + fname, out)  
   

input_scaler = preprocessing.StandardScaler().fit(train_input)
output_scaler = preprocessing.StandardScaler().fit(train_output)
train_input = input_scaler.transform(train_input)
train_output = output_scaler.transform(train_output)

def train_model():

   global model
   # Create the model	
   model = Sequential()

   # INPUT LAYER
   model.add(Dropout(0.0, input_shape=(inp_dim,)))
   model.add(Dense(inp_dim,activation='selu'))

   # HIDDEN 1
   model.add(Dense(encoding_dim,  activation='selu'))
   #model.add(Dropout(0.2))

   # HIDDEN 2
   model.add(Dense(encoding_dim,  activation='selu'))
   #model.add(Dropout(0.2))

   model.add(Dense(out_dim,  activation='selu'))

   # Compile the model
   sgd = SGD(lr=0.1, momentum=0.2, decay=1e-6, nesterov=False)
   model.compile(optimizer=sgd, loss=custom_mse)
   model.summary()
   model.fit(train_input,train_output,epochs=4, batch_size=32, shuffle=True,callbacks=[LoggingCallback(logging.info)])

train_model()
test_model()


