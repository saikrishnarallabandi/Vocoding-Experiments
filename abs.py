import scipy, pylab
import soundfile as sf
from scipy.io import wavfile as wf
import numpy as np
import librosa

def stft_librosa(y, fs, n_fft=2048, win_length=1024, hop_length=512, n_mels=None, power=1 ):

    linear = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    mag = np.abs(linear) # magnitude 

    if n_mels is not None:
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels) # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag**power) # (n_mels, t) # mel spectrogram
    else:
        mel = None

    return linear, mag, mel

def griffinlim(spectrogram, n_iter=50, n_fft=2048, win_length=2048, hop_length=512):
 
    print("Shape of spectrogram is ", spectrogram.shape)
   
    def invert_spectrogram(spectrogram):
         return librosa.istft(spectrogram, hop_length, win_length)

    import copy
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y
 
 
def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

def main():
   
  fs,A = wf.read('LA_D_1110070.wav')
  print(len(A)/fs)

  ############ Baseline
  X = stft(A,fs, 0.01, 0.005)
  x_hat = istft(X, fs, int(len(A)/fs), 0.005)
  ############

  ############ Librosa
  A, fs = sf.read('LA_D_1110070.wav')
  print(A)
  (_, X, _) = stft_librosa(A, fs)
  x_hat = griffinlim(X)
  sf.write('abs.wav', np.asarray(x_hat), 16000,format='wav',subtype="PCM_16")     
  
main()
   
