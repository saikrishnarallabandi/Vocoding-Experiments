#
# file located in same directory as feats
#
voice=slt
mkdir -p feats scripts 
cd feats

# Download the features
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/feats_${voice}_1msec.tar.gz
tar xvzf feats_${voice}_1msec.tar.gz

# Download pitchmarks and wave files
cd ../
mkdir -p voices/cmu_us_${voice}
cd voices/cmu_us_${voice}
$FESTVOXDIR/src/setup_cg cmu us $[voice}
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/pm_$[voice}.tar.gz
tar xvzf pm_$[voice}.tar.gz
wget http://festvox.org/cmu_arctic/packed/cmu_us_$[voice}_arctic.tar.bz2
tar xvjf cmu_us_$[voice}_arctic.tar.bz2
mv cmu_us_$[voice}_arctic/wav .
cd ../

# If we dont have pitch marks, make them
cd ../
mkdir -p voices/cmu_us_$[voice}
cd voices/cmu_us_$[voice}
wget http://festvox.org/cmu_arctic/packed/cmu_us_$[voice}_arctic.tar.bz2
tar xvjf cmu_us_$[voice}_arctic.tar.bz2
mv cmu_us_$[voice}_arctic/wav .
./bin/make_pm_wave wav/* # If female voice, we need to change the arguments
./bin/make_pm_fix pm/*
cd ../

# Clone the repo
mkdir -p repos
cd repos
git clone https://github.com/saikrishnarallabandi/Vocoding-Experiments
cd ..

# Get into the kitchen
mkdir -p kitchen_$[voice}
# copy 02_wav_*, test_synth* initial_map* util.py model.py
# Set paths and folder names correctly
# Run 
