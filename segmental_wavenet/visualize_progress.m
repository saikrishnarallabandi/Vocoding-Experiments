% Clear the workspace
clc; clear all; close all;

% Definitions
log_dir = '/home/srallaba/Documents/projects/vocoders/segmental_wavenet/kitchen/train_eval';

% Find all the png files
files = dir(strcat(log_dir, '/*.png'));

% Iterate over them, obtain the target and prediction
for i = 1 :length(files)
    
     file = files(i).name;
     [str,tok] = strtok(file, '_');
     
     target_filename = strcat(log_dir , '/', str, '_target.wav');
     predicted_filename = strcat(log_dir, '/' , str, '_predicted.wav');
     
     [target,fs] = audioread(target_filename);
     [predicted,fs] = audioread(predicted_filename);


     subplot(2,1,1);
     plot(target); hold off ; 
     title(strcat("Visualizing ", str));

     subplot(2,1,2);
     plot(predicted); hold off;     

     pause;
     
    
end


