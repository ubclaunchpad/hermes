from warpctc_pytorch import CTCLoss
from data.SpectroSequenceDataset import SpectroSequenceDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.CTCRNNModel import CTCRNNModel

# Select the proper device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = SpectroSequenceDataset('valid_test.h5')
print("\nDataset loading completed\n")
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

model = CTCRNNModel(input_dim, hidden_dim, output_dim)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ctc_loss = CTCLoss()
count = 0
print("Begin training")
for epoch in range(10):
    for seq, label in dataset:
        model.zero_grad()

        # Get the distributions
        seq = torch.tensor([seq], device = device)
        log_probs = model(seq.float())
        log_probs = log_probs.transpose(0, 1).cuda().contiguous()
        log_probs.requires_grad_(True)

        # Construct the labels
        label =  label.replace(" ", "")
        label = torch.IntTensor([char_to_ix[char] for char in label])

        # Batch size info for CTC
        probs_sizes = torch.IntTensor([log_probs.shape[0]])
        label_sizes = torch.IntTensor([label.shape[0]])

        cost = ctc_loss(log_probs, label, probs_sizes, label_sizes)

        count += 1
        if (count % 10 == 0):
            print(cost)

        # Backprop, update gradients
        cost.backward()
        optimizer.step()
