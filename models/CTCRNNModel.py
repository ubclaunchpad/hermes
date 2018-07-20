import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CTCRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTCRNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = 2, batch_first = True, bidirectional = True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2alphabet = nn.Linear(in_features = hidden_dim * 2, out_features = output_dim)
        #self.hidden = self.init_hidden()
        #self.cell = self.init_cell()

    #def init_hidden(self):
    #    return torch.zeros(2, 1, self.hidden_dim)

    #def init_cell(self):
    #    return torch.zeros(2, 1, self.hidden_dim)

    def forward(self, input_sequence):
        #lstm_out, self.hidden, self.cell = self.lstm(input_sequence, self.hidden, self.cell)
        lstm_out, _, = self.lstm(input_sequence)

        tag_space = self.hidden2alphabet(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores
