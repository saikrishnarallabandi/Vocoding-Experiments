import os, sys
import numpy as np
import numpy as np
import os,sys
from sklearn import preprocessing
import pickle, logging
from sklearn.metrics import mean_squared_error
import dynet as dy
import time, random

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
hidden = 512


test_dir = exp_dir + '/test'
resynth_dir = exp_dir + '/resynth'
logfile_name = exp_dir + '/logs/log' 

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
    if '9' in input_file:
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
num_valid = len(valid_files)
valid_data = zip(valid_input, valid_output, valid_files)
random.shuffle(valid_data)
test_data = valid_data[0:int(num_valid/2)]
valid_data = valid_data[int(num_valid)/2+1:]


class FeedForwardNeuralNet(object):

  def __init__(self, model, args):
    self.pc = model.add_subcollection()
    self.args = args
    self.num_input = int(args[0])
    self.num_output = int(args[2])
    self.hidden_list = args[1]
    self.act = args[3]
    self.model = model
    self.number_of_layers = len(self.hidden_list)
    num_hidden_1 = self.hidden_list[0]
    
    # Add first layer
    self.W1 = self.pc.add_parameters((num_hidden_1, self.num_input))
    self.b1 = self.pc.add_parameters((num_hidden_1))
    
    # Add remaining layers
    self.weight_matrix_array = []
    self.biases_array = []
    self.weight_matrix_array.append(self.W1)
    self.biases_array.append(self.b1)
    for k in range(1, self.number_of_layers):
              self.weight_matrix_array.append(self.model.add_parameters((self.hidden_list[k], self.hidden_list[k-1])))
              self.biases_array.append(self.model.add_parameters((self.hidden_list[k])))
    self.weight_matrix_array.append(self.model.add_parameters((self.num_output, self.hidden_list[-1])))
    self.biases_array.append(self.model.add_parameters((self.num_output)))
    self.spec = (self.num_input, self.hidden_list, self.num_output, self.act)
   
  def basic_affine(self, exp):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)
    return dy.tanh(dy.affine_transform([b1,W1,exp]))

  def calculate_loss(self, input, output):
    #dy.renew_cg()
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b)) 
    acts = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    act = acts[0]
    intermediate = act(dy.affine_transform([b, w, input]))
    activations = [intermediate]
    for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
        pred = g(dy.affine_transform([b, W, activations[-1]]))
        activations.append(pred)  
    #print output.value(), pred.value()
    losses = output - pred
    return dy.l2_norm(losses)

  def predict(self, input):
    weight_matrix_array = []
    biases_array = []
    acts = []
    for (W,b, act) in zip(self.weight_matrix_array, self.biases_array, self.act):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b))
         acts.append(act)
    g = acts[0]
    w = weight_matrix_array[0]
    b = biases_array[0]
    intermediate = g(w*input + b)
    activations = [intermediate]
    for (W,b, act) in zip(weight_matrix_array[1:], biases_array[1:], acts):
        pred =  act(W * activations[-1]  + b)
        activations.append(pred)
    return pred

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_list, num_output, act = spec
    return FeedForwardNeuralNet(model, [num_input, hidden_list, num_output, act])

print valid_data

def test_model():
   # Test each file
   for (inp, out, fname) in valid_data:
       print inp
       inp = input_scaler.transform(inp)
       pred = []
       for input_frame in inp:
           input_frame = dy.inputTensor(input_frame)
           pred_frame = dnn.predict(input_frame)
           pred.append(np.asarray(pred_frame.value()))
       pred = np.asarray(pred)
       pred = output_scaler.inverse_transform(pred)
       np.savetxt(test_dir + '/' + fname, pred)  
       np.savetxt(resynth_dir + '/' + fname, out)
       print "Saved in ", test_dir
   
   for (inp, out, fname) in test_data:
       inp = input_scaler.transform(inp)
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
train_input = input_scaler.transform(train_input)
train_output = output_scaler.transform(train_output)

# Hyperparameters 
units_input = inp_dim
units_hidden = int(hidden)
units_output = out_dim

global m
m = dy.Model()
dnn = FeedForwardNeuralNet(m, [units_input, [units_hidden, units_hidden, units_hidden, units_hidden], units_output, [dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh]])
trainer = dy.SimpleSGDTrainer(m)
update_params = 32
num_epochs = 4


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
        count = 1
        K += 1
        dy.renew_cg()
        frame_count += 1
        count += 1
        loss = dnn.calculate_loss(dy.inputTensor(i), dy.inputTensor(o))
        train_loss += loss.value()
        loss.backward()
        if frame_count % int(0.1*num_train) == 1:
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


