import sys
import os
from os import path
from glob import glob
import numpy as np
import pandas as pd
from sox import Transformer
from progressbar import ProgressBar

pbar = ProgressBar()

# define variables
testing_data_dir = path.abspath('cv_corpus_v1/cv-valid-test/waves')
training_data_dir = path.abspath('cv_corpus_v1/cv-valid-train/waves')
sample_rate = 16000

# first generate wav files using command line, cd into the directory containing
# the mp3 files and run the following command:
# for i in *.mp3
# do
#     lame --decode "$i" "waves/$(basename -s .mp3 "$i").wav"
# done
# this will put the corresponding wav files in a new directory called waves

def generate_waves(data_dir):
    # change audio file to 16k sample rate
    for wav_file in pbar(glob(path.join(data_dir, '*.wav'))):
        new_file = path.splitext(wav_file)[0] + "k16.wav"
        transformer = Transformer()
        transformer.convert(samplerate= sample_rate)
        transformer.build(wav_file, new_file)

    # remove old files
    for item in pbar(glob(path.join(data_dir, '*.wav'))):
        if item.endswith("k16.wav"):
            continue
        else:
            os.remove(item)

    # rename files to remove k16
    for item in pbar(glob(path.join(data_dir, '*.wav'))):
        os.rename(item, item.replace('k16', ''))

# generate waves for testing and training set
generate_waves(testing_data_dir)
generate_waves(training_data_dir)
