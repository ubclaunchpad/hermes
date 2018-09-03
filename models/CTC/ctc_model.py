import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(CTCModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # The gru takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = 4, bidirectional = True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2alphabet = nn.Linear(in_features = hidden_dim * 2, out_features = output_dim)

        self.hidden = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(8, batch_size, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True).cuda()

    def forward(self, X, X_lengths):
        batch_size, seq_len, _ = X.size()
        input_sequences = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        output_sequences, _ = self.gru(input_sequences, self.hidden)
        output_sequences, _ = torch.nn.utils.rnn.pad_packed_sequence(output_sequences, batch_first=True)
        tag_space = self.hidden2alphabet(output_sequences)
        return tag_space
