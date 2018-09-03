from warpctc_pytorch import CTCLoss
from data.spectrogram_dataset import SpectrogramDataset, merge_batches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.CTC.ctc_model import CTCModel
from torch.utils.data import DataLoader
# Select the proper device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5')
data_loader = DataLoader(dataset, collate_fn = merge_batches, batch_size = 4, shuffle = True)
print("dataset len")
print(dataset.__len__())
print("\nDataset loading completed\n")

# Dimention of FFTs
input_dim = 128

# Dimention of hidden state
hidden_dim = 512

# Alphabet size with a blank
output_dim = 30
# Well
learning_rate = 1e-4

batch_size = 4

model = CTCModel(input_dim, hidden_dim, output_dim, batch_size)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

ctc_loss = CTCLoss(blank=0)
count = 0
print("Begin training")
for epoch in range(50):
    print("***************************")
    print("EPOCH NUM %d" % epoch)
    print("***************************")
    cost_epoch_sum = 0
    cost_tstep_sum = 0
    for i_batch, sample_batched in enumerate(data_loader):
        optimizer.zero_grad()
        padded_X, seq_labels, X_lengths, Y_lengths = sample_batched
        # Get the distributions
        padded_X = padded_X.cuda()
        log_probs = model(padded_X, X_lengths)
        log_probs = log_probs.transpose(0, 1)

        log_probs.requires_grad_(True)

        seq_labels = torch.cat(seq_labels)

        cost = ctc_loss(log_probs, seq_labels, X_lengths, Y_lengths)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        optimizer.step()
        print(cost)
        # Backprop, update gradients
    print("***************************")
    print("AVG COST PER EPOCH")
    print(cost_epoch_sum / 4076)
    print("***************************")
