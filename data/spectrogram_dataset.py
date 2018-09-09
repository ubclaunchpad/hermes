import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class SpectrogramDataset(Dataset):
    """
        PyTorch dataset for fetching data batches from data stored in HDF5 format
    """

    def __init__(self, hdf5_location, model_ctc = False):
        self.char_to_ix = {'a' : 0, 'b' : 1, 'c' : 2, 'd':  3,
                            'e': 4, 'f': 5, 'g': 6, 'h':7, 'i':8, 'j': 9, 'k': 10,
                            'l': 11, 'm' : 12, 'n' : 13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19,
                            'u' : 20, 'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24, 'z' : 25, "'" : 26, " ": 27, "_": 28}
        self.hdf5 = h5py.File(hdf5_location, 'r')
        self.model_ctc = model_ctc
        self.myidx = 0

    def __len__(self):
        return len(self.hdf5["train_data"])

    def __getitem__(self, idx):
        flat = np.concatenate(self.hdf5["train_data"][idx],  axis = 0)
        return (flat.reshape(-1, 128), self.hdf5["train_labels"][idx])

    def set_transform(self, transform):
        self.transform = transform

    def merge_batches(self, batch):
        """
            Stacks multiple sequences into a minibatch suitable for training
        """
        batch_size = len(batch)
        X_lengths = np.zeros(batch_size, dtype = "int")
        Y_lengths = np.zeros(batch_size, dtype = "int")
        # Find lengths of feature and label sequences
        for i in range(batch_size):
            x_length, y_length = len(batch[i][0]), len(batch[i][1])
            X_lengths[i] = x_length
            Y_lengths[i] = y_length
        longest_seq_x = max(X_lengths)
        # In descending orderFloatTensor
        X_seq_indices = np.argsort(-X_lengths)
        # Batch is represented as batch_size x longest_sequence x feature_dim
        padded_X = torch.zeros((batch_size, longest_seq_x, 128)).type(torch.FloatTensor)
        seq_labels = [0] * batch_size
        # copy over the actual sequences
        for i, seq_num in enumerate(X_seq_indices):
            sequence, label = torch.FloatTensor(batch[seq_num][0]), batch[seq_num][1]
            label = torch.IntTensor([self.char_to_ix.get(char, 28) for char in label])
            x_len, y_len = X_lengths[seq_num], Y_lengths[seq_num]
            sequence = self.transform(sequence)
            padded_X[i, 0:x_len, :] = sequence[:x_len, :]
            seq_labels[i] = label
        seq_labels = torch.cat(seq_labels)
        if (self.model_ctc == True):
            return padded_X, seq_labels, torch.IntTensor(X_lengths[X_seq_indices]), torch.IntTensor(Y_lengths[X_seq_indices])
        # https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106
        # On keeping track of the association between two sequences after they have been reordered in the batch
        # by decreasing order
        Y_seq_indices = np.argsort(-Y_lengths)
        pad_token = - 1
        longest_sent = max(Y_lengths) + 1
        embedding_dim = len(self.char_to_ix)
        # Batch size x Timesteps + 1 x Characters in alphabet
        padded_Y = np.ones((batch_size, longest_sent, embedding_dim)) * pad_token
        onehot_vecs = torch.eye(embedding_dim)
        for i, seq_num in enumerate(Y_seq_indices):
            label = batch[seq_num][1]
            label = torch.stack([onehot_vecs[self.char_to_ix.get(char, 28)] for char in label])
            y_len = Y_lengths[seq_num]
            # First vector in the sequence needs to be zeros
            padded_Y[i, 1:y_len + 1, :] = label[:y_len, :]
            padded_Y[i, 0, :] = 0
        indices = (X_seq_indices, Y_seq_indices)
        lengths = (torch.IntTensor(X_lengths[X_seq_indices]), torch.IntTensor(Y_lengths[Y_seq_indices]))
        return padded_X, torch.FloatTensor(padded_Y), seq_labels, indices, lengths

class Normalize(object):

    def __init__(self, dataset, samples = 100):
        stacked = []
        for i in range(samples):
            stacked += [k for k in dataset[i][0]]
        self.mu = torch.FloatTensor(np.mean(stacked, axis=0))
        self.std = torch.FloatTensor(np.std(stacked, axis=0))

    def __call__(self, sample):
        return (sample - self.mu)/self.std
