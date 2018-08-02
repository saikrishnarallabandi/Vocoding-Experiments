import os
import numpy as np
from scipy.io import wavfile as wf
import soundfile as sf
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt


'''
Read pitchmark file
Get 25 msec window around it ( if possible)
Read ccoeffs file and get the ccoeffs.
Read the wavefile and get corresponding wave.
Quantize the wave and store <ccoeffs,wav> in numpy array. 
'''

save_plot = 0

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
   # Scipy and soundfile read wavefiles differently
   fs,A = wf.read(file)
   A,fs = sf.read(file)
   return A

   x_1 = (A / 32768.0).astype(np.float32)
   y_1 = mulaw_quantize(x_1,256)   
   return y_1


def make_pm_ccoeffs_wav_pm(file):

  pitchmark_file = file
  timestamp_array = read_pmfile(pitchmark_file)
  fname = os.path.basename(file).split('.')[0]

  ccoeffs_file = '../feats/slt_arctic/' + fname + '.ccoeffs_ascii'
  ccoeffs = np.loadtxt(ccoeffs_file)

  wav_file = '../voices/cmu_us_slt/wav/' + fname + '.wav'
  wav_quantized = quantize_wavfile(wav_file)


  for i, t in enumerate(timestamp_array):
    period_start = float(t) - 0.005 if float(t) - 0.005 > 0 else 0
    period_end = float(t) + 0.005 if float(t) + 0.005 < timestamp_array[-1] else timestamp_array[-1]
    frame_start = int(period_start * 1000)
    frame_end = int(period_end * 1000)
    sample_start = int(period_start * 16000)
    sample_end = int(period_end * 16000)
    
    #print period_start, period_end, frame_start, frame_end, sample_start, sample_end
    #print len(ccoeffs), len(wav_quantized)
    #print fname + '-audio-' + str(i).zfill(4) + '.npy|' + fname + '-mel-' + str(i).zfill(4) + '.npy|' + str(sample_end-sample_start) +  '|N/A|0'    

    c = ccoeffs[frame_start: frame_end]
    np.save("data_new/" + fname + '-mel-' + str(i).zfill(4) + '.npy', c)
    w = wav_quantized[sample_start:sample_end]
    np.save("data_new/" + fname + '-audio-' + str(i).zfill(4) + '.npy',w)
    if len(w) == 160 and len(c) == 10:   
        g.write(fname + '-audio-' + str(i).zfill(4) + '.npy|' + fname + '-mel-' + str(i).zfill(4) + '.npy|' + str(sample_end-sample_start) +  '|N/A|0' + '\n')
        h.write(fname + ' ' + str(sample_end-sample_start) + ' ' + str(frame_end - frame_start) + '\n') 
        if save_plot :
          axes = plt.gca()
          axes.set_ylim([-0.9,0.9])
          plt.plot(w)
          plt.savefig('data_new/' + fname + '-plot-' + str(i).zfill(4) + '_noquantization.png')
          plt.close()

pitchmark_files = sorted(os.listdir('../voices/cmu_us_slt/pm/'))
g = open('train.txt','w')
h = open('log.txt','w')
l = 0
for f in pitchmark_files:
 #print f, l
 #if l < 2:
  if f[0] == '.':
   continue
  else: 
   print "Processing ", f
   filename = '../voices/cmu_us_slt/pm/' + f
   make_pm_ccoeffs_wav_pm(filename)
   l += 1
g.close()
h.close()
