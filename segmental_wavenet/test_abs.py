import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.io import wavfile as wf
import librosa
from matplotlib import pyplot as plt
import soundfile as sf

filename = 'arctic_a0001'
folder = 'data_new'
files = sorted(os.listdir(folder))
wave_array = []
for file in files:
   if filename in file and 'audio' in file:
      A = np.load(folder + '/' + file)
      for a in A:
          wave_array.append(a)

w = np.array(wave_array) * 1.0 #/ 256.0 # * 32000
print len(w), w[0:10], np.max(w)
librosa.output.write_wav('00_noquantization.wav', np.asarray(w), 16000)
sf.write('00_noquantization.wav', np.asarray(w), 16000,format='wav',subtype="PCM_16")     

plt.plot(w)
plt.savefig('00_noquantization.png')


fs,A = wf.read('../voices/cmu_us_slt/wav/arctic_a0001.wav')
plt.close()
plt.plot(A)
plt.savefig('orig_wav.png')
