import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw

# Locations
data_dir = '../wav'
feats_dir = '../feats_world'

if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)

files = sorted(os.listdir(data_dir))

for file in files:
    fname = file.split('.')[0]
    print fname
    x, fs = sf.read(data_dir + '/' + file)
    f0, sp, ap = pw.wav2world(x, fs) 
    np.savetxt(feats_dir + '/' + fname + '.f0_ascii', f0)
    np.savetxt(feats_dir + '/' + fname + '.sp_ascii', sp)    
    np.savetxt(feats_dir + '/' + fname + '.ap_ascii', ap)    
