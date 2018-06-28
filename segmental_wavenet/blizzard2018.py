  from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import cmu_arctic
from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa
import sys

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw
from hparams import hparams

linguistic_feats_path = '/home2/srallaba/challenges/blizzard2018/data/input_files_npy/'
ccoeffs_feats_path = 'ccoefs_ascii/'

g = open('logfile','w')
g.close()

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = cmu_arctic.available_speakers
    speakers = ['awb']

    wd = cmu_arctic.WavFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    speaker_ids = wd.labels

    for index, (speaker_id, wav_path) in enumerate(
            zip(speaker_ids, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, "N/A")))
    return [future.result() for future in tqdm(futures)]


def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False

def ensure_divisible(sig, length):
    l = len(sig)
    if l % length == 0:
         return sig
    else:
       difference = l % length
       #for k in range(difference):
       #    np.append(sig, sig[-1])
       return sig[:len(sig)-difference]

def ensure_frameperiod(sig, mel):
   length = mel.shape[0]
   l = len(sig)
   g = open('logfile','a')
   g.write('Original length of signal: ' + str(len(sig)) + ' original number of frames ' + str(mel.shape[0]) + '\n')

   if float(80 * length) == l:
      g.write('Not changing anything ' + '\n')
      g.close()
      return sig, mel
   else:
      num_samples = 80 * length
      if num_samples > l:
         difference = int((num_samples - l))
         g.write("Signal is shorter"+ '\n')
         g.close()
         for k in range(difference):
           sig =  np.append(sig, sig[-1])
         return sig, mel

      elif num_samples < l:
         difference = int((l - num_samples))
         g.write("Signal is longer" + '\n')
         g.close()
         return sig[:len(sig)-difference], mel
         
      else:
         print("This is hard")
         sys.exit()         
     
 

def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate

    # Load the audio to a numpy array. Resampled if needed
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start:end]
        out = out[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32


    #print("Wavepath is ", wav_path)
    filename = wav_path.split('/wav/')[-1].split('.wav')[0]
    fname = filename
    filename = ccoeffs_feats_path + '/' + filename + '.mcep'
    mel_spectrogram = np.loadtxt(filename)
    #print("Shape of mel scptrogram is ", mel_spectrogram.shape)
    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    #mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    #l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    #out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    #out = ensure_divisible(out, N)
    #print("Length of out: ", len(out), "N ", N)

    #print("Out and N: ", len(out), N)
    #if len(out) < N * audio.get_hop_size():
        #print("Out and N: ", filename, len(out), N, N * audio.get_hop_size())   
    #    sys.exit()
    #assert len(out) >= N * audio.get_hop_size()
   
    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    #out = out[:N * 80]
    #out = ensure_divisible(out, N)
    g = open('logfile','a')
    g.write("Processing " + fname + '\n')
    g.close()
   
    out,mel_spectrogram = ensure_frameperiod(out,mel_spectrogram)
    #out = ensure_divisible(out, audio.get_hop_size())
    #assert len(out) % audio.get_hop_size() == 0
    #assert len(out) % N == 0
    timesteps = len(out)
    g = open('logfile','a')
    g.write(fname + ' ' + str(len(out)) + ' ' + str(N) + ' ' + str(len(out) % N) + '\n')
    g.write('\n')
    g.close()

    # Write the spectrograms to disk:
    audio_filename = fname + '-audio-%05d.npy' % index
    mel_filename = fname + '-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, timesteps, text, speaker_id)
