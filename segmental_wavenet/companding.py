from scipy.io import wavfile as wf
import numpy as np

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

def make_equal_lengths(a,b):
    len_a = len(a)
    len_b = len(b)
    min_len = min(len_a, len_b)
    return a[:min_len], b[:min_len] 

file = 'amidsummernightsdream_0001.wav'
fs,A = wf.read(file)
x_1 = (A / 32768.0).astype(np.float32)
y_1 = mulaw_quantize(x_1,256)
y = 2 * _asfloat(y_1) / 256 - 1
y_resynth = inv_mulaw(y,256)
wf.write('temp.wav', fs, y_resynth)

file = 'The_Leopard_and_the_Sky_God_00000_00001_00001.wav'
fs,B = wf.read(file)
x_2 = (B / 32768.0).astype(np.float32)
x_2 = B
y_2 = mulaw_quantize(x_2,256)
y_2 -= 5
print(len(y_2))
#y_2 = y_1
y = 2 * _asfloat(y_2) / 256 - 1
y_resynth = inv_mulaw(y,256)
wf.write('temp_wavenet.wav', fs, y_resynth)


'''


a,b = make_equal_lengths(y_2, y_1)
c = a + b
y = 2 * _asfloat(c) / 256 - 1
y_resynth = inv_mulaw(y,512)
wf.write('temp_RF_mean.wav', fs, y_resynth)






import keras
from keras.models import Sequential
from keras.layers import Dense, AlphaDropout
from keras.callbacks import *
import pickle, logging
from keras.utils import to_categorical


a,b = make_equal_lengths(y_2, y_1)
y = 2 * _asfloat(a) / 256 - 1
y_resynth = inv_mulaw(y,256)
wf.write('temp_RF_keras_lengthed.wav', fs, y_resynth)


a,b = to_categorical(a,256), to_categorical(b,256)
x_train = np.array_split(a, 16)
y_train = np.array_split(b, 16)




hidden = 128
global model
model = Sequential()
model.add(Dense(256, kernel_initializer='lecun_normal', activation='relu', input_shape=(256,)))
model.add(Dense(hidden, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dense(hidden, kernel_initializer='lecun_normal', activation='relu'))
model.add(Dense(256, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()
model.fit(a,b, batch_size=16, epochs=5, shuffle=True)


a_predicted = np.argmax(model.predict(a),axis=-1)
print a_predicted[0:20]
print y_2[0:20]
y = 2 * _asfloat(a_predicted) / 256 - 1
y_resynth = inv_mulaw(y,256)
wf.write('temp_RF_keras.wav', fs, y_resynth)


'''
