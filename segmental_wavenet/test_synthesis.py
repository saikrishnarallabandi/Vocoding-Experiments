import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
from scipy.io import wavfile as wf
#import librosa
from matplotlib import pyplot as plt
import soundfile as sf
from scipy.signal import hann

filename = sys.argv[1]
epoch = sys.argv[2]
print ( "Synthesizing " , filename) 
file = '../voices/cmu_us_slt/wav/' + filename + '.wav'
A,fs = sf.read(file)
A_abs = np.zeros(len(A))
noise = np.random.normal(0,0.001,len(A))
#A_abs += noise

def read_pmfile(file):
   f = open(file)
   lines = f.readlines()
   timestamp_array = []
   for i, line in enumerate(lines):        
       #print line  
       if i > 9:
          pitch_mark = line.split('\n')[0].split()[0]
          timestamp_array.append(pitch_mark)
   return timestamp_array

pitchmark_file = '../voices/cmu_us_slt/pm/' + filename + '.pm'
timestamp_array = read_pmfile(pitchmark_file)
folder = 'data_synthesis'
files = sorted(os.listdir(folder))

for i, t in enumerate(timestamp_array):
 try:
   a = np.load(folder + '/' + filename + '-audio-' + str(i).zfill(4) + '.npy')    # arctic_a0001-audio-0000.npy
   period_start = float(t) - 0.005 if float(t) - 0.005 > 0 else 0
   period_end = float(t) + 0.005 if float(t) + 0.005 < timestamp_array[-1] else timestamp_array[-1]
   frame_start = int(period_start * 1000)
   frame_end = int(period_end * 1000)
   sample_start = int(period_start * 16000)
   sample_end = int(period_end * 16000)
   window = hann(len(a), sym=False)
   #print i, t, sample_start, sample_end, sample_end - sample_start, len(a), len(A) #, a[20:40]
   if sample_end > len(A):
      continue
   if sample_end - sample_start == 160 and len(a) == 160:
      A_abs[sample_start:sample_end] += a * window  # Direct add
      
   elif sample_end - sample_start == 161 and len(a) == 160:
      A_abs[sample_start:sample_end-1] += a * window
   elif sample_end - sample_start == 159 and len(a) == 160:
      A_abs[sample_start:sample_end+1] += a * window
   elif sample_end - sample_start == 160 and len(a) == 161:
      window = hann(len(a)-1, sym=False)
      A_abs[sample_start:sample_end] += a[1:] * window
   elif sample_end - sample_start == 161 and len(a) == 161:
      window = hann(len(a)-1, sym=False)
      A_abs[sample_start:sample_end-1] += a[1:] * window
   elif sample_end - sample_start == 159 and len(a) == 159:
      window = hann(len(a)-1, sym=False)
      A_abs[sample_start:sample_end-1] += a[1:] * window
   elif len(a) == 161:
      window = hann(len(a), sym=False)
      A_abs[sample_start:sample_end] += a[1:] * window
   elif len(a) == 159:
      A_abs[sample_start:sample_end-1] += a * window
   else:
      pass
   #A_abs[sample_start:sample_end] = A_abs[sample_start:sample_end] / 2.0
 except IOError:
   continue
 
sf.write('abs_1msec_' + filename + '_epoch' + str(epoch).zfill(3) + '.wav', np.asarray(A_abs), 16000,format='wav',subtype="PCM_16")           



'''


wave_array = []
for i, file in enumerate(files):
   if filename in file and 'audio' in file:
      print file
      A = np.load(folder + '/' + file)
      for a in A:
          wave_array.append(a)

w = np.array(wave_array) * 1.0 #/ 256.0 # * 32000
print len(w), w[0:10], np.max(w)
#librosa.output.write_wav('abs_1msec_' + filename + '.wav', np.asarray(w), 16000)
sf.write('abs_1msec_' + filename + '.wav', np.asarray(w), 16000,format='wav',subtype="PCM_16")     

plt.plot(w)
plt.savefig('abs_1msec_' + filename + '.png')


'''
