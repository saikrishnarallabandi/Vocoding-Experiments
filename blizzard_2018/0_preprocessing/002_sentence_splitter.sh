#!/usr/bin/bash

wavfiles_dir='/home/saikrishnalticmu/data/blizzard_release_2017/enUK/fls/'
labfiles_dir='/home/saikrishnalticmu/Downloads/blizzard_training_data.git'
cwd=`pwd`

for folder in $wavfiles_dir/*
  do

  	echo "In " $folder
    story=$(basename "$folder") 
    story=${story,,}

    rm -rf $folder/lab
    cd $folder
    echo cp -r $labfiles_dir/$story/lab $folder/lab/
    cp -r $labfiles_dir/$story/lab $folder/lab/ 

    cd $folder/lab
    # Remove spaces from labs
    for f in *\ *; do echo $f; mv "$f" "${f// /_}"; done
    echo "Removed the spaces from labs"

    cd $folder/wav 
    # Remove spaces from wavs
    for f in *\ *; do echo $f; mv "$f" "${f// /_}"; done
    echo "Removed the spaces from wavs"  

    cd ${cwd}
     
    echo python make_labs_wavs.py $folder/wav $folder/lab $story
    python make_labs_wavs.py $folder/wav $folder/lab $story
    echo 
    echo  
  done




