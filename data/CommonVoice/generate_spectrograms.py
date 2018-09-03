import os
from os import path
from glob import glob
import numpy as np
import pandas as pd
from sox import Transformer
import progressbar
from progressbar import ProgressBar
import h5py
import soundfile as sf
import pydub
import librosa.display



# define variables
#testing_data_dir = path.abspath('cv_corpus_v1/cv-valid-test/wavs')
#training_data_dir = path.abspath('cv_corpus_v1/cv-valid-train/wavs')

training_data_dir = "/media/grigorii/External/cv-valid-train"

sample_rate = 16000

def generate_wavs(data_dir):
    print("hello")
    pbar = ProgressBar()
    for mp3_file in pbar(glob(path.join(data_dir, '*.mp3'))):
        sound = pydub.AudioSegment.from_mp3(mp3_file)
        filename = mp3_file[-10:-4]
        new_file = path.splitext(data_dir)[0] +"/wavs/" + filename + ".wav"
        sound.export(new_file, format="wav")
    pbar = ProgressBar()
    data_dir = data_dir + '/wavs/'
    # change audio file to 16k sample rate
    for wav_file in pbar(glob(path.join(data_dir, '*.wav'))):
        new_file = path.splitext(wav_file)[0] + "k16.wav"
        transformer = Transformer()
        transformer.convert(samplerate= sample_rate)
        transformer.build(wav_file, new_file)
    pbar = ProgressBar()
    # remove old files
    for item in pbar(glob(path.join(data_dir, '*.wav'))):
        if item.endswith("k16.wav"):
            continue
        else:
            os.remove(item)
    pbar = ProgressBar()
    # rename files to remove k16
    for item in pbar(glob(path.join(data_dir, '*.wav'))):
        os.rename(item, item.replace('k16', ''))
    print("end")

# generate waves for testing and training set
#generate_wavs(testing_data_dir)
generate_wavs(training_data_dir)


# define variables
#testing_data_dir = path.abspath('cv_corpus_v1/cv-valid-test/wavs-test')
#testing_csv = 'cv_corpus_v1/cv-valid-test.csv'
training_data_dir = "/media/grigorii/External/cv-valid-train/wavs"
#training_csv = 'cv_corpus_v1/cv-valid-train.csv'
training_csv = "/media/grigorii/External/cv-valid-train.csv"
sample_rate = 16000


def generate_spectrograms(data_dir, csv_file, output, features_dataset, labels_dataset):
    # read csv file
    df = pd.read_csv(
        csv_file,
        sep=',',
        usecols=[0, 1],
        names=['filename', 'text'],
        engine='python'
    )

    # h5 file to store dataset
    with h5py.File(output, 'w') as file:
        paths = glob(path.join(data_dir, '*.wav'))
        dt = h5py.special_dtype(vlen=np.dtype('float64'))
        features = file.create_dataset(features_dataset, (len(paths), 128, ), dtype=dt)
        dt = h5py.special_dtype(vlen=str)
        labels = file.create_dataset(labels_dataset, (len(paths),), dtype=dt)

        pbar = ProgressBar(max_value=progressbar.UnknownLength)
        # read audio files
        print('current set: ' + data_dir)
        for i, wav_file in enumerate(paths):
            # read signal and sample_rate from wav file
            index = int(wav_file[-10:-4])
            print(wav_file)
            print(index)
            signal, sample_rate = sf.read(wav_file)
            # generate chroma spectogram using params
            chroma = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft = 400, hop_length = 160)
            # save into h5 file
            features[index] = chroma
            labels[index] = df['text'][index + 1]
            pbar.update(i)

#get spectograms for testing and training data
#generate_spectrograms(testing_data_dir, testing_csv, 'valid_test.h5', 'test_data', 'test_labels')
generate_spectrograms(training_data_dir, training_csv, '/media/grigorii/External/valid_train.h5', 'train_data', 'train_labels')
