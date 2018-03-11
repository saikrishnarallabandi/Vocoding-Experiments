#!/bin/bash

LANG=C; export LANG


# Clone the repo
git clone https://github.com/mmorise/World
cd World
make || exit 0
export WORLD_DIR=`pwd`/build 
cd examples/analysis_synthesis/
make
cd $WORLD_DIR/..

# Get a sample wav
wget http://tts.speech.cs.cmu.edu/rsk/tts_stuff/conversational_wav/20100109-067-000.wav
mkdir -p test_installation
mv 20100109-067-000.wav test_installation

# Analysis
cd test_installation
$WORLD_DIR/analysis 20100109-067-000.wav test.f0 test.sp test.ap

# Synthesis
$WORLD_DIR/synthesis test.f0 test.sp test.ap test.wav
