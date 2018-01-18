Build an MVP first.

wget the following link:
http://tts.speech.cs.cmu.edu/rsk/tts_stuff/tts_mvp_DNNWORLD_18Jan2018.tar.gz

Once extracted, it creates two directories: data and scripts
If you navigate to scripts and hit run.sh, all should work.
It generates a directory '../expts/MVP' and stores intermediate results like log file, wavefiles, etc in that
from the main directory,

expt/MVP/logs - contains the log file
expt/MVP/test - contains the test wavefile synthesized based on NN predictions
expt/MVP/resynth - contains the same test wavefile but resynthesized based on original features ( for quality comparision)
expt/MVP/models - stores models ( for future use. doesnt contain anything now)
