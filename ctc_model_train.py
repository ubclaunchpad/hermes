from warpctc_pytorch import CTCLoss
from data.SpectroSequenceDataset import SpectroSequenceDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.CTCRNNModel import CTCRNNModel


dataset = SpectroSequenceDataset('valid_test.h5')

char_to_ix = {' ': 0, 'a' : 1, 'b' : 2, 'c' : 3, 'd':  4,
             'e': 5, 'f': 6, 'g': 7, 'h':8, 'i':9, 'j': 10, 'k': 11,
             'l': 12, 'm' : 13, 'n' : 14, 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20,
             'u' : 21, 'v' : 22, 'w' : 23, 'x' : 24, 'y' : 25, 'z' : 26, ',': 27, "'" : 28}


# Dimention of FFTs
input_dim = 128

# Dimention of hidden state
hidden_dim = 128

# Alphabet size with a blank
output_dim = 29
# Well
learning_rate = 0.01
# Dimention of each element of the sequence of output activations (distribution for 26 alphabet letters plus blank)
label_sizes = torch.IntTensor([output_dim])

model = CTCRNNModel(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ctc_loss = CTCLoss()

for epoch in range(100):
    for seq, label in dataset:

        model.zero_grad()
        batch_wrapper = [0]
        batch_wrapper[0] = seq
        seq = torch.tensor(batch_wrapper)
        log_probs = model(seq.float())
        log_probs = log_probs.transpose(0, 1)
        label = torch.IntTensor([char_to_ix[char] for char in label])
        probs_sizes = torch.IntTensor([log_probs.shape[0]])
        log_probs.requires_grad_(True)
        cost = ctc_loss(log_probs, label, probs_sizes, label_sizes)
        cost.backward()
