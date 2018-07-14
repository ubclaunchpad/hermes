import sys
import os
from os import path
from glob import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
from progressbar import ProgressBar

pbar = ProgressBar()

# define variables
testing_data_dir = path.abspath('cv_corpus_v1/cv-valid-test/waves')
testing_csv = 'cv_corpus_v1/cv-valid-test.csv'
training_data_dir = path.abspath('cv_corpus_v1/cv-valid-train/waves')
training_csv = 'cv_corpus_v1/cv-valid-train.csv'
sample_rate = 16000

def generate_spectrograms(data_dir, csv_file, output):
    # read csv file
    df = pd.read_csv(
        csv_file,
        sep=',',
        usecols=[0, 1],
        names=['filename', 'text'],
        engine='python'
    )
    print(df.head())

    # read audio files
    X = []
    Y = []

    for wav_file in pbar(glob(path.join(data_dir, '*.wav'))):
        # read signal and sample_rate from wav file
        index = int(wav_file[-10:-4]) + 1
        signal, sample_rate = sf.read(wav_file)
        # generate chroma spectogram using params
        chroma = librosa.feature.melspectrogram(y=signal, sr=sample_rate,
               n_fft = 400, hop_length = 160)
        chroma_t = chroma.T
        X.append(chroma_t)
        Y.append(df['text'][index])

    dd.io.save(output, (X, Y))

# get spectograms for testing and training data
generate_spectrograms(testing_data_dir, testing_csv, 'valid_test.h5')
generate_spectrograms(training_data_dir, training_csv, 'valid_train.h5')
