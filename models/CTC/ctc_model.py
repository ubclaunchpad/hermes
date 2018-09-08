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
        self.conv1 = torch.nn.Conv2d(1, 32,  (1, 3), stride=(1, 2), padding = 0)
        self.conv2 = torch.nn.Conv2d(32, 32, (1, 6), stride=(1, 2), padding = 0)
        self.conv3 = torch.nn.Conv2d(32, 4, (1, 6), stride=(1, 2), padding = 0)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size = 48, hidden_size = hidden_dim, num_layers = 2, bidirectional = True, batch_first = True)
        self.dp = nn.Dropout(p = 0.4)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2alphabet = nn.Linear(in_features = hidden_dim * 2, out_features = output_dim)
        self.hidden = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(4, batch_size, self.hidden_dim).cuda()), requires_grad=True).cuda()

    def forward(self, X, X_lengths = [], train = True):
        if (len(X_lengths) == 0):
            X.unsqueeze_(1)
            print(X.shape)
            X = self.conv1(X)
            X = self.relu(X)
            X = self.conv2(X)
            X = self.relu(X)
            X = self.conv3(X)
            X = torch.transpose(X, 1, 2)
            print(X.size())
            batch, time, filter, feat = X.size()
            X = X.contiguous().view(1, time, filter * feat)
            print(X.shape)
            output_sequences, _ = self.gru(X, self.hidden[:, :1, :].contiguous())
            tag_space = self.hidden2alphabet(output_sequences)
            if (not train):
                return tag_space
            softmax = F.log_softmax(tag_space, dim = 2)
            return softmax
        else:
            batch_size, seq_len, _ = X.size()
            X.unsqueeze_(1)
            X = self.conv1(X)
            X = self.relu(X)
            X = self.dp(X)
            X = self.conv2(X)
            X = self.relu(X)
            X = self.conv3(X)
            X = torch.transpose(X, 1, 2)
            filter, feat = X.shape[2], X.shape[3]
            X = X.contiguous().view(batch_size, seq_len, filter * feat)
            input_sequences = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
            output_sequences, _ = self.gru(input_sequences, self.hidden)
            output_sequences, _ = torch.nn.utils.rnn.pad_packed_sequence(output_sequences, batch_first=True)
            tag_space = self.hidden2alphabet(output_sequences)
            return tag_space
