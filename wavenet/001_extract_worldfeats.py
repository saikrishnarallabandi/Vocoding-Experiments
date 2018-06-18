import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw
import gzip
import pickle

# Locations
data_dir = '/home2/srallaba/projects/tts/segmental_wavenet/data/wav_5887_16k'
feats_dir = '../feats'

if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)

files = sorted(os.listdir(data_dir))

for file in files:
    fname = file.split('.')[0]
    print fname
    if os.path.exists(feats_dir + '/' + fname + '.f0_ascii.gz'):
       print "File exists!! Skipping"
       continue
    x, fs = sf.read(data_dir + '/' + file)
    f0, sp, ap = pw.wav2world(x, fs) 

    
    # Compress using gunzip
    with gzip.open(feats_dir + '/' + fname + '.ap_ascii.gz', 'wb') as f:
        f.write(ap)
    print "Size of the ap array is ", os.path.getsize(feats_dir + '/' + fname + '.ap_ascii.gz')

    with gzip.open(feats_dir + '/' + fname + '.f0_ascii.gz', 'wb') as f:
        f.write(f0)
    print "Size of the ap array is ", os.path.getsize(feats_dir + '/' + fname + '.ap_ascii.gz')

    with gzip.open(feats_dir + '/' + fname + '.sp_ascii.gz', 'wb') as f:
        f.write(sp)
    print "Size of the ap array is ", os.path.getsize(feats_dir + '/' + fname + '.ap_ascii.gz')
    

