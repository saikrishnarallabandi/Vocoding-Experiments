import os
import numpy as np
from scipy.io import wavfile as wf
import matplotlib
import sys
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import soundfile as sf

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
       if i > 9:
          pitch_mark = line.split('\n')[0].split()[0]
          timestamp_array.append(pitch_mark)
   return timestamp_array

def quantize_wavfile(file):
   A,fs = sf.read(file)
   x_1 = (A / 32768.0).astype(np.float32)
   y_1 = mulaw_quantize(x_1,256)   
   return y_1


def make_pm_ccoeffs_wav_pm(file):

  pitchmark_file = file
  timestamp_array = read_pmfile(pitchmark_file)
  fname = os.path.basename(file).split('.')[0]

  ccoeffs_file = '../feats/slt_arctic_1msec/' + fname + '.ccoeffs_ascii'
  ccoeffs = np.loadtxt(ccoeffs_file)

  wav_file = '../voices/cmu_us_slt/wav/' + fname + '.wav'
  wav_quantized = quantize_wavfile(wav_file)


  for i, t in enumerate(timestamp_array):
    period_start = float(t) - 0.005 if float(t) - 0.005 > 0 else 0
    period_end = float(t) + 0.005 if float(t) + 0.005 < timestamp_array[-1] else timestamp_array[-1]
    frame_start = int(float( period_start * 1000)/ 1) # 1 msec
    frame_end = int(float( period_end * 1000)/ 1) # 1 msec
    sample_start = int(period_start * 16000)
    sample_end = int(period_end * 16000)

    c = ccoeffs[frame_start: frame_end]
    np.save("data_new/" + fname + '-mel-' + str(i).zfill(4) + '.npy', c)
    w = wav_quantized[sample_start:sample_end]
    np.save("data_new/" + fname + '-audio-' + str(i).zfill(4) + '.npy',w)

    axes = plt.gca()
    axes.set_ylim([-0.9,0.9])
    plt.plot(w)
    plt.savefig('data_new/' + fname + '-plot-' + str(i).zfill(4) + '_noquantization.png')
    plt.close()

    #print i, t

    
    if len(w) == 160 and len(c) == 10:   
        g.write(fname + '-audio-' + str(i).zfill(4) + '.npy|' + fname + '-mel-' + str(i).zfill(4) + '.npy|' + str(sample_end-sample_start) +  '|N/A|0' + '\n')
        h.write(fname + ' ' + str(sample_end-sample_start) + ' ' + str(frame_end - frame_start) + '\n') 
    else:
       #print "This is not happening. The length of w is  ", len(w), " and that of c is ", len(c)
       #print i
       pass
       #sys.exit(0)
    '''
    # Just bypass the conditional and try to see if abs is better
    g.write(fname + '-audio-' + str(i).zfill(4) + '.npy|' + fname + '-mel-' + str(i).zfill(4) + '.npy|' + str(sample_end-sample_start) +  '|N/A|0' + '\n')
    h.write(fname + ' ' + str(sample_end-sample_start) + ' ' + str(frame_end - frame_start) + '\n')
    '''


pitchmark_files = sorted(os.listdir('../voices/cmu_us_slt/pm/'))
if not os.path.exists('data_new'):
    os.makedirs('data_new')

g = open('train.txt','w')
h = open('log.txt','w')
l = 0
for f in pitchmark_files:
 print f, l
 if l < 10000: 
  if f[0] == '.':
   print "Ignoring"
   print '\n'
   continue
  elif f.endswith('.wav.pm'): 
   print "Ignoring"
   print '\n'
   continue
  else:
   print "Processing ", f
   filename = '../voices/cmu_us_slt/pm/' + f
   make_pm_ccoeffs_wav_pm(filename)
   l += 1
   #cmd = "python2 test_abs_v2.py " + f.split('.pm')[0] 
   #os.system(cmd)
   print '\n'

g.close()
h.close()
