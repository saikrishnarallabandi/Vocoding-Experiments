import numpy as np
import os

def get_sequences(arr, input_dim=56, max_len=100):
   '''
   We need a function that takes any array of elements and converted them into sequences. 
   For this, we should already know the maximum length of the sequences and obviously the dimensions of the elements. 
   We should also know the length of the array.
   '''
   length_array = len(arr)

   seq = np.zeros((length_array, max_len, input_dim), dtype='float32')
   
   for i, s in enumerate(arr):
      length = len(s)
      print s
      kk = np.zeros((length_array-length,input_dim))     
      seq[i] = np.concatenate((s, kk),axis=0)
 
   return seq


def text_to_sequences():
   '''
   Take the text, corresponding f0 sequence as input and output arrays.
   '''

   text_file = '/home2/srallaba/data/tts_stuff_lstm/txt.done.data.temp'
   f0_folder = '/home2/srallaba/data/tts_stuff_lstm/f0_ascii'

   # Store text and f0 as  list of lists. Outer list contains the utterances. Inner list contains the characters, f0 in each utterance. 
   text_array = []
   f0_array = []

   f = open(text_file)
   for line in f:
      line = line.split('\n')[0].split()
      fname = line[0]
      char_array = []
      for c in line[1:]:
          for ch in c:
             char_array.append(ch)
          char_array.append(' ')
      text_array.append(char_array)
      f0_utterance = np.loadtxt(f0_folder + '/' + fname + '.f0')
      f0_array.append(f0_utterance)

   return text_array, f0_array   
