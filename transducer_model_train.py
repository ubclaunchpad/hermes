from data.spectrogram_dataset import SpectrogramDataset, Normalize

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.transducer.transducer_model import TransducerModel
from models.transducer.transducer_decoder import RNNTransducer
import models.transducer.transducer_awni.functions.transducer as transducer
from torch.utils.data import DataLoader
import numpy as np
from progressbar import ProgressBar
# Select the proper device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_transducer(rnn_layers, learning_rate):
    dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5')
    norm_transform = Normalize(dataset)
    decoder = RNNTransducer(dataset.char_to_ix)
    dataset.set_transform(norm_transform)
    batch_size = 8

    data_loader = DataLoader(dataset, collate_fn = dataset.merge_batches, batch_size = batch_size, shuffle = True)
    print("dataset len")
    print(dataset.__len__())
    print("\nDataset loading completed\n")

    # Dimention of FFTs
    input_dim = 128

    # Dimention of hidden state
    hidden_dim = 256

    # Alphabet size with a blank
    output_dim = 30

    model = TransducerModel(input_dim, hidden_dim, output_dim, rnn_layers, batch_size)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    transducer_loss = transducer.TransducerLoss(blank_label = 0)
    count = 0
    print("Begin training")
    for epoch in range(60):
        print("***************************")
        print("EPOCH NUM %d" % epoch)
        print("***************************")
        cost_epoch_sum = 0
        cost_tstep_sum = 0
        pbar = ProgressBar()
        for sample_batched in pbar(data_loader):
            optimizer.zero_grad()
            padded_X, padded_Y, seq_labels, indices, lengths = sample_batched
            X_lengths, Y_lengths = lengths
            if (X_lengths[0] > 2500):
                continue
            X_lengths = (X_lengths - 6) // 2
            lengths = (X_lengths, Y_lengths)
            if (len(X_lengths) < batch_size):
                break
            # Get the distributions
            padded_X = padded_X.cuda()
            padded_Y = padded_Y.cuda()

            prob_matrix, Y_lengths = model(padded_X, padded_Y, indices, lengths)
            prob_matrix = prob_matrix.contiguous()
            prob_matrix.requires_grad_(True)
            cost = transducer_loss(prob_matrix, seq_labels, X_lengths, Y_lengths)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 300)
            optimizer.step()
            #print(cost)
            cost_epoch_sum += float(cost)
            # Backprop, update gradients

        # TODO: Decoding
        print("Avg cost per epoch: ", cost_epoch_sum / 50000)
        """
        print("***************************")
        print("PREDICTION")
        xseq, yseq = dataset[0]
        xseq = torch.FloatTensor([xseq], device = device)
        xseq = norm_transform(xseq)
        print(type(xseq))
        print(xseq)
        log_probs = model(xseq.float().cuda())
        logprobs_numpy = log_probs[0].data.cpu().numpy()
        for row in logprobs_numpy:
            print(row)
        decoded_seq, _ = decoder.beam_search_decoding(log_probs[0].data.cpu().numpy(), beam_size = 100)
        print("Ground truth: ", yseq)
        print("Prediction: ", decoded_seq)
        print(decoded_seq[0])
        print("***************************")
        """
while(True):
    learning_rates = [1e-4, 1e-3]
    num_rnn_layers = [2, 3]
    for rnn_layers in num_rnn_layers:
        for learning_rate in learning_rates:
            #while True:
            #    try:
            train_transducer(rnn_layers, learning_rate)
            #        break
            #    except Exception:
            #        print("caught nan")

"""
def train_transducer():
    dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5')
    norm_transform = Normalize(dataset)
    decoder = RNNTransducer(dataset.char_to_ix)
    dataset.set_transform(norm_transform)

    batch_size = 1

    data_loader = DataLoader(dataset, collate_fn = dataset.merge_batches, batch_size = batch_size, shuffle = True)
    print("dataset len")
    print(dataset.__len__())
    print("\nDataset loading completed\n")

    # Dimention of FFTs
    input_dim = 128

    # Dimention of hidden state
    hidden_dim = 256

    # Alphabet size with a blank
    output_dim = 30

    learning_rate = 1e-2

    model = TransducerModel(input_dim, hidden_dim, output_dim, batch_size)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    transducer_loss = transducer.TransducerLoss(blank_label = 0)
    count = 0
    print("Begin training")
    for epoch in range(50):
        print("***************************")
        print("EPOCH NUM %d" % epoch)
        print("***************************")
        cost_epoch_sum = 0
        cost_tstep_sum = 0
        for i in range(500):
            x, y = dataset[0]
            len_label = len(y)
            len_x = torch.IntTensor([x.shape[0]])
            optimizer.zero_grad()
            # Get the distributions
            embedding_dim = len(dataset.char_to_ix)
            # Batch size x Timesteps + 1 x Characters in alphabet
            padded_Y = np.ones((1, len_label + 1, embedding_dim))
            onehot_vecs = torch.eye(embedding_dim)

            label = torch.stack([onehot_vecs[dataset.char_to_ix.get(char, 28)] for char in y])
            # First vector in the sequence needs to be zeros
            padded_Y[0, 1:len_label + 1, :] = label[:len_label, :]
            padded_Y[0, 0, :] = 0
            x = torch.FloatTensor([x]).cuda()
            padded_Y = torch.FloatTensor(padded_Y).cuda()

            prob_matrix, _ = model(x, padded_Y)
            prob_matrix.requires_grad_(True)
            len_label = torch.IntTensor([len_label])
            seq_label = torch.IntTensor([dataset.char_to_ix.get(char, 28) for char in y])
            cost = transducer_loss(prob_matrix, seq_label, len_x, len_label)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 300)
            optimizer.step()
            print(cost)

train_transducer()
"""
