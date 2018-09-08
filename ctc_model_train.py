from warpctc_pytorch import CTCLoss
from data.spectrogram_dataset import SpectrogramDataset, Normalize

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.CTC.ctc_model import CTCModel
from models.CTC.ctc_decoder import CTCDecoder
from torch.utils.data import DataLoader
# Select the proper device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 4

def train_ctc():
    dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5', model_ctc = True)
    norm_transform = Normalize(dataset)
    decoder = CTCDecoder(dataset.char_to_ix)
    dataset.set_transform(norm_transform)
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

    learning_rate = 1e-3

    model = CTCModel(input_dim, hidden_dim, output_dim, batch_size)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    ctc_loss = CTCLoss(blank = output_dim - 1)
    count = 0
    print("Begin training")
    for epoch in range(100):
        print("***************************")
        print("EPOCH NUM %d" % epoch)
        print("***************************")
        cost_epoch_sum = 0
        cost_tstep_sum = 0
        for i_batch, sample_batched in enumerate(data_loader):
            optimizer.zero_grad()
            padded_X, seq_labels, X_lengths, Y_lengths = sample_batched
            if (len(X_lengths) < batch_size):
                break
            # Get the distributions
            padded_X = padded_X.cuda()
            log_probs = model(padded_X, X_lengths)
            log_probs = log_probs.transpose(0, 1)
            log_probs.requires_grad_(True)
            cost = ctc_loss(log_probs, seq_labels, X_lengths, Y_lengths)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 300)
            optimizer.step()
            print(cost)
            cost_epoch_sum += float(cost)

        print("***************************")
        print("PREDICTION")
        model = model.eval()
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
        model = model.train()
        print("Ground truth: ", yseq)
        print("Prediction: ", decoded_seq)
        #print("Avg cost per epoch: ", cost_epoch_sum / 4076)
        print("***************************")


"""
def train_ctc():
    dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5', model_ctc = True)
    norm_transform = Normalize(dataset)
    decoder = CTCDecoder(dataset.char_to_ix)
    dataset.set_transform(norm_transform)
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

    learning_rate = 1e-3

    model = CTCModel(input_dim, hidden_dim, output_dim, batch_size)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ctc_loss = CTCLoss(blank=0)
    count = 0
    print("Begin training")
    for epoch in range(50):
        print("***************************")
        print("EPOCH NUM %d" % epoch)
        print("***************************")
        cost_epoch_sum = 0
        cost_tstep_sum = 0
        for i in range(500):
            optimizer.zero_grad()
            xseq, yseq = dataset[0]
            xseq = torch.FloatTensor([xseq], device = device)
            xseq = norm_transform(xseq)
            log_probs = model(xseq.float().cuda())
            log_probs = log_probs.transpose(0, 1)
            label = torch.IntTensor([dataset.char_to_ix[char] for char in yseq])
            log_probs.requires_grad_(True)
            probs_sizes = torch.IntTensor([log_probs.shape[0]])
            label_sizes = torch.IntTensor([label.shape[0]])
            cost = ctc_loss(log_probs, label, probs_sizes, label_sizes)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 600)
            optimizer.step()
            print(cost)
            # Backprop, update gradients
        print("***************************")
        print("PREDICTION")
        xseq, yseq = dataset[0]
        xseq = torch.FloatTensor([xseq], device = device)
        xseq = norm_transform(xseq)
        print(type(xseq))
        print(xseq)
        log_probs = model(xseq.float().cuda(), train = False)
        logprobs_numpy = log_probs[0].data.cpu().numpy()
        for row in logprobs_numpy:
            print(row)
        decoded_seq, _ = decoder.beam_search_decoding(log_probs[0].data.cpu().numpy(), beam_size = 100)
        print("Ground truth: ", yseq)
        print("Prediction: ", decoded_seq)
        print(decoded_seq[0])
        print("***************************")
"""
train_ctc()
