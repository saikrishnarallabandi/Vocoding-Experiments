
# Folder structure
mkdir segmental_wavenet
cd segmental_wavenet
mkdir -p voices data scripts repos kitchen

# Download the feats
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/feats_slt_5msec_25June2018.tar.gz
tar xvzf feats_slt_5msec_25June2018.tar.gz 

# Download the pitchmarks and wavefiles for slt
cd voices
mkdir cmu_us_slt
cd cmu_us_slt
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/pm_slt_26June2018.tar.gz
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/data/wav_slt_26June2018.tar.gz
tar xvzf pm_slt_26June2018.tar.gz
tar xvzf wav_slt_26June2018.tar.gz
cd ../..

# Download repos
# We will use an adapted version of r9y9's toolkit for experiments
https://github.com/r9y9/wavenet_vocoder
cd wavenet_vocoder
pip install -e ".[train]"
rm -r train.py
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/kitchen/segmental-wavenet-experiments/train.py
cd ..
git clone https://github.com/saikrishnarallabandi/Vocoding-Experiments
cd ../scripts

# Prepare data
cp ../repos/Vocoding-Experiments/segmental_wavenet/02_make_ccoeffs_wav_pm.py .
mkdir data
python 02_make_ccoeffs_wav_pm.py

cd ../repos/wavenet_vocoder
cp ../../scripts/train.txt ../../scripts/data/
python3.5 train.py --data-root=../../scripts/data
