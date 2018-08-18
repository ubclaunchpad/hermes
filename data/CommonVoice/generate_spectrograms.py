import sys
import os
from os import path
from glob import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import h5py
from progressbar import ProgressBar

pbar = ProgressBar()

# define variables
testing_data_dir = path.abspath('cv_corpus_v1/cv-valid-test/waves-test')
testing_csv = 'cv_corpus_v1/cv-valid-test.csv'
training_data_dir = path.abspath('cv_corpus_v1/cv-valid-train/waves')
training_csv = 'cv_corpus_v1/cv-valid-train.csv'
sample_rate = 16000


def generate_spectrograms(data_dir, csv_file, output, data_group, labels_group):
    # read csv file
    df = pd.read_csv(
        csv_file,
        sep=',',
        usecols=[0, 1],
        names=['filename', 'text'],
        engine='python'
    )
    print(df.head())

    # h5 file to store dataset
    file = h5py.File(output,'w')
    data = file.create_group(data_group)
    labels = file.create_group(labels_group)

    # read audio files
    print('current set: ' + data_dir)
    for wav_file in pbar(glob(path.join(data_dir, '*.wav'))):
        # read signal and sample_rate from wav file
        index = int(wav_file[-10:-4])
        signal, sample_rate = sf.read(wav_file)
        # generate chroma spectogram using params
        chroma = librosa.feature.melspectrogram(y=signal, sr=sample_rate,
               n_fft = 400, hop_length = 160)
        chroma_t = chroma.T
        # save into h5 file
        data.create_dataset(str(index), data=chroma_t)
        labels.create_dataset(str(index), data=df['text'][index + 1])

# get spectograms for testing and training data
generate_spectrograms(testing_data_dir, testing_csv, 'valid_test.h5', 'test_data', 'test_labels')
generate_spectrograms(training_data_dir, training_csv, 'valid_train.h5', 'train_data', 'train_labels')
