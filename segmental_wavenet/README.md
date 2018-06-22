nohup ./do_world wav2world ../voices/cmu_us_slt/wav/ filenames ../feats/slt_arctic/ > log_ccoeffs_slt_1msec 2>&1&


*For larger datasets, it might be better to do file by file and save space* 


./do_world wav2world_file ../data/LJSpeech-1.0/wavs/ filenames ../feats/slt_arctic/
