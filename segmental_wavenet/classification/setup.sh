cd feats

# Download the features
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/feats_slt_1msec_25June2018.tar.gz
tar xvzf feats_slt_1msec_25June2018.tar.gz

# Download pitchmark files
cd ../
mkdir -p voices/cmu_us_slt
cd voices/cmu_us_slt
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/pm_slt_26June2018.tar.gz
tar xvzf pm_slt_26June2018.tar.gz

# Download wav files
wget http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2
tar xvjf cmu_us_slt_arctic.tar.bz2
mv cmu_us_slt_arctic/wav .
cd ../

# Clone the repo
mkdir -p repos
cd repos
git clone https://github.com/saikrishnarallabandi/Vocoding-Experiments
cd ..

# Get into the kitchen
mkdir -p kitchen
