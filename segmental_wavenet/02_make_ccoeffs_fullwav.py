import os
import numpy as np
from scipy.io import wavfile as wf

'''
Read pitchmark file
Get 25 msec window around it ( if possible)
Read ccoeffs file and get the ccoeffs.
Read the wavefile and get corresponding wave.
Quantize the wave and store <ccoeffs,wav> in numpy array. 
'''

def mulaw(x, mu=256):
   return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)

def mulaw_quantize(x, mu=256):
   y = mulaw(x, mu)
   return _asint((y + 1) / 2 * mu)

def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def inv_mulaw(y, mu=256):
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def read_pmfile(file):
   f = open(file)
   lines = f.readlines()
   timestamp_array = []
   for i, line in enumerate(lines):        
       if i > 7:
          pitch_mark = line.split('\n')[0].split()[0]
          timestamp_array.append(pitch_mark)
   return timestamp_array

def quantize_wavfile(file):
   fs,A = wf.read(file)
   #return A

   x_1 = (A / np.max(A)).astype(np.float32)
   return x_1

   x_1 = (A / 32000).astype(np.float32)
   return x_1


   x_1 = A
   y_1 = mulaw_quantize(x_1,256)   
   return y_1

   return x_1

def make_ccoeffs_wav(file):

  fname = os.path.basename(file).split('.')[0]

  ccoeffs_file = '../feats/slt_arctic_5msec/' + fname + '.ccoeffs_ascii'
  ccoeffs = np.loadtxt(ccoeffs_file)

  wav_file = '../voices/cmu_us_slt/wav/' + fname + '.wav'
  wav_quantized = quantize_wavfile(wav_file)
  print "The maximum qunatizated value in this file is ", np.max(wav_quantized)
  print '\n'

  i = len(ccoeffs) -1 # 22 July 2018 Sai Krishna It is always better to cut than to append
  frame_start = 0
  frame_end = i 
  sample_start = int(i * 80) # 80 because 5msec. 
  sample_end = int((i+10) * 80)

  c = ccoeffs[frame_start: frame_end]
  np.save("data_fullwav/" + fname + '-mel-' + str(i).zfill(4) + '.npy', c)
  w = wav_quantized[sample_start:sample_end]
  np.save("data_fullwav/" + fname + '-audio-' + str(i).zfill(4) + '.npy',w)
    
    #print "Frames from ", frame_start, frame_end, " of " , len(ccoeffs), " | Waves from ",  sample_start, sample_end, " of ", len(wav_quantized)
  if len(w) == 800 and len(c) == 10:   
        g.write(fname + '-audio-' + str(i).zfill(4) + '.npy|' + fname + '-mel-' + str(i).zfill(4) + '.npy|' + str(sample_end-sample_start) +  '|N/A|0' + '\n')
        h.write(fname + ' ' + str(sample_end-sample_start) + ' ' + str(frame_end - frame_start) + '\n') 
    

pitchmark_files = sorted(os.listdir('../voices/cmu_us_slt/pm/'))
g = open('train.txt','w')
h = open('log.txt','w')
for f in pitchmark_files:
 if f[0] == '.':
   continue
 else: 
   print "Processing ", f
   filename = '../voices/cmu_us_slt/pm/' + f
   make_ccoeffs_wav(filename)
g.close()
h.close()
