#
# file located in same directory as feats
#

cd feats

# Download the features
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/feats_slt_1msec_25June2018.tar.gz
tar xvzf feats_slt_1msec_25June2018.tar.gz

# Download pitchmarks and wave files
cd ../
mkdir -p voices/cmu_us_slt
cd voices/cmu_us_slt
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/pm_slt_26June2018.tar.gz
tar xvzf pm_slt_26June2018.tar.gz
wget http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2
tar xvjf cmu_us_slt_arctic.tar.bz2
mv cmu_us_slt_arctic/wav .
cd ../

# If we dont have pitch marks, make them
cd ../
mkdir -p voices/cmu_us_slt
cd voices/cmu_us_slt
wget http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2
tar xvjf cmu_us_slt_arctic.tar.bz2
mv cmu_us_slt_arctic/wav .
./bin/make_pm_wave wav/* # If female voice, we need to change the arguments
./bin/make_pm_fix pm/*
cd ../

# Clone the repo
mkdir -p repos
cd repos
git clone https://github.com/saikrishnarallabandi/Vocoding-Experiments
cd ..

# Get into the kitchen
mkdir -p kitchen_slt
# copy 02_wav_*, test_synth* initial_map* util.py model.py
# Set paths and folder names correctly
# Run 
