import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw

fs = int(sys.argv[1])
f0_file = sys.argv[2]
sp_file = sys.argv[3]
ap_file = sys.argv[4]
dest = sys.argv[5]

# Locations

f0 = np.loadtxt(f0_file)
sp = np.loadtxt(sp_file)
ap = np.loadtxt(ap_file)
y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
sf.write(dest, y, fs)

