import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TransducerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(TransducerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.transcription_gru = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = 2, bidirectional = True, batch_first = True)
        self.prediction_gru = nn.GRU(input_size = 29, hidden_size = hidden_dim, num_layers = 2, batch_first = True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2density_transcript = nn.Linear(in_features = hidden_dim * 2, out_features = hidden_dim)
        self.hidden2density_pred = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.relu = nn.ReLU()
        self.density2softmax = nn.Linear(in_features = hidden_dim, out_features = output_dim)
        # Hidden states
        self.hidden_trascription = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(4, batch_size, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True).cuda()
        self.hidden_prediction = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2, batch_size, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True).cuda()

    def forward(self, X, Y, indices = (), lengths = ()):
        if (indices == ()):
            transcript_dist = self.transcription_net(X)
            predict_dist = self.prediction_net(Y)
            predict_dist = predict_dist.unsqueeze(dim = 1)
            transcript_dist = transcript_dist.unsqueeze(dim = 2)
            prob_density = predict_dist + transcript_dist
            prob_density = self.relu(prob_density)
            prob_density = self.density2softmax(prob_density)
            prob_density_normalized = F.log_softmax(prob_density, dim = 3)
            return prob_density_normalized, ()

        X_seq_indices, Y_seq_indices = indices
        X_lengths, Y_lengths = lengths
        # Run transcription network
        batch_size, T, _ = X.size()
        transcript_dist = self.transcription_net(X, X_lengths)
        # Run prediction network
        predict_dist = self.prediction_net(Y, Y_lengths)
        inverse_permut = np.argsort(Y_seq_indices)
        # Reorder the sequences in predict_dist (in batch dimention)
        # to the order of sequences in transcript_dist
        predict_dist = predict_dist[inverse_permut][X_seq_indices]
        Y_lengths = Y_lengths[inverse_permut][X_seq_indices]
        # Batch x T x U x Alphabet
        predict_dist = predict_dist.unsqueeze(dim = 1)
        transcript_dist = transcript_dist.unsqueeze(dim = 2)
        prob_density = predict_dist + transcript_dist
        prob_density = self.relu(prob_density)
        prob_density = self.density2softmax(prob_density)
        prob_density_normalized = F.log_softmax(prob_density, dim = 3)
        return prob_density_normalized, Y_lengths

    def transcription_net(self, X, X_lengths = []):
        if (len(X_lengths) == 0):
            out_transcript, _ = self.transcription_gru(X, self.hidden_trascription)
            transcript_dist = self.hidden2density_transcript(out_transcript)
            return transcript_dist
        in_ffts = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        out_transcript, _ = self.transcription_gru(in_ffts, self.hidden_trascription)
        out_transcript, _ = torch.nn.utils.rnn.pad_packed_sequence(out_transcript, batch_first=True)
        # Should be Batch x T x Alphabet
        transcript_dist = self.hidden2density_transcript(out_transcript)
        return transcript_dist

    def prediction_net(self, Y, Y_lengths = []):
        if (len(Y_lengths) == 0):
            out_prediction, _ = self.prediction_gru(Y, self.hidden_prediction)
            predict_dist = self.hidden2density_pred(out_prediction)
            return predict_dist
        Y_lengths_0 = Y_lengths + 1
        in_labels = torch.nn.utils.rnn.pack_padded_sequence(Y, Y_lengths_0, batch_first=True)
        out_prediction, _ = self.prediction_gru(in_labels, self.hidden_prediction)
        out_prediction, _ = torch.nn.utils.rnn.pad_packed_sequence(out_prediction, batch_first=True)
        # Should be Batch x U x Alphabet
        predict_dist = self.hidden2density_pred(out_prediction)
        return predict_dist

    def infer():
        """
            Predict the label
        """
        encode()
        pass
