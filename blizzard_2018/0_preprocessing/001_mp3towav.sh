 #!/usr/bin/bash

files_dir='/home/saikrishnalticmu/data/blizzard_release_2017/enUK/fls'
wav_dir='/home/saikrishnalticmu/Projects/Blizzard_2018/wav'
rm -r $wav_dir
mkdir $wav_dir

cd ${files_dir}
START_DIR=${files_dir}

for folder in *;
   do
    
    echo "I am currently in"
    echo        ${folder}

    echo "Removing " ${folder}/wav 
    rm -r ${folder}/wav 
    mkdir -p ${folder}/wav  

    #detox $START_DIR/$folder/"audio"
    pacpl --to wav -r -p ${folder}/audio --outdir ${folder}/wav
    cd $START_DIR/$folder/"wav";
    
    # Remove spaces
    for f in *\ *; do echo $f; mv "$f" "${f// /_}"; done
    echo "Removed the spaces" 

    # Copy the files
    for file in *.wav ; do echo "Copying " $file " to " $wav_dir/${folder}_$file; cp $file $wav_dir/${folder}_$file; done  
    
    
    cd $START_DIR


   done

