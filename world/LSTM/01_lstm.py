import numpy as np
from tensorflow.python.keras.utils import to_categorical
from collections import defaultdict
import dynet as dy
import random

wids = defaultdict(lambda: len(wids))
wids["</s>"] = 0
text_file = '/home3/srallaba/data/tts_stuff_lstm/txt.done.data.temp'
lim = 50 
i = 0
texts_array = []
feats_array = []
f = open(text_file)
for line in f:
  if i < lim:
    i += 1
    line = line.split('\n')[0]
    fname = line[1:13]
    print fname
    text = line[15:]
    charray = []
    for d in text:
       charray.append(str(wids[d]))
    texts_array.append(charray)
    feats = np.loadtxt('/home3/srallaba/data/tts_stuff/output_full/' + fname + '.ccoeffs')
    feats_array.append(feats)

'''    
i2w = {i:w for w,i in wids.iteritems()}
num_char = len(wids)
input_dim = num_char
max_len_text = get_max_len(texts_array)
train_input = get_sequences_categorical(texts_array, input_dim, max_len_text)

out_dim = 66
max_len_feats = get_max_len(feats_array)
train_output = get_sequences(feats_array, out_dim, max_len_feats)
'''

num_char = len(wids)
print "Number of characters: ", num_char
train_data = zip(texts_array, feats_array)

embedding_size = 32
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)
encoder_lstm = dy.VanillaLSTMBuilder(1, embedding_size, 32,model)
decoder_lstm = dy.VanillaLSTMBuilder(1, embedding_size, 32,model)
lookup = model.add_lookup_parameters((num_char, embedding_size))
decoder_weight = model.add_parameters((66,32))
decoder_bias = model.add_parameters((66))

for epoch in range(20):
   random.shuffle(train_data)
   i = 1
   train_loss = 0
   for (t,f) in train_data:
      dy.renew_cg()
      # Encode the char sequence
      s = encoder_lstm.initial_state()
      for char in t:
           s.add_input(dy.lookup(lookup, int(char)))
                           
      # Decode the feat sequence
      dec_init_state = decoder_lstm.initial_state()
      losses = []
      last_output_embedding = dy.lookup(lookup, 0)
      s = dec_init_state.add_input(last_output_embedding)
      #s = dec_init_state.add_input(dy.concatenate([dy.vecInput(64), last_output_embedding]))
      idx = 0
      W = dy.parameter(decoder_weight)
      b = dy.parameter(decoder_bias)
      idx = 0
      while True:
         #print idx
         idx += 1
         if idx > 10000:
              break
         for feat in f:
             #print idx, feat
             last_output_embedding = s.output()  
             pred = dy.affine_transform([b, W, last_output_embedding])
             losses.append(dy.squared_norm(pred - dy.inputTensor(feat)))
             if len(losses) > 50:
                    break
      loss = dy.esum(losses)
      train_loss += loss.value()
      loss.backward()
      trainer.update()
   print "Train loss : ", train_loss 


