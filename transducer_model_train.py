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
    dataset = SpectrogramDataset('/media/grigorii/External/valid_train.h5')
    norm_transform = Normalize(dataset)
    decoder = RNNTransducer(dataset.char_to_ix)
    dataset.set_transform(norm_transform)
    batch_size = 4

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
    checkpoint = torch.load("/home/grigorii/model_dicts/transducer_epoch_42.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded, begin training")
    for epoch in range(32, 60):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()
            #print(cost)
            cost_epoch_sum += float(cost)
            # Backprop, update gradients
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': cost_epoch_sum / 34000,
            }, "/home/grigorii/model_dicts/transducer_epoch_%d.pt" % epoch)
        # TODO: Decoding
        print("Avg cost per epoch: ", cost_epoch_sum / 34000)
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

while(True):A[y][0],
#    learning_rates = [1e-4, 1e-3]
#    num_rnn_layers = [2, 3]
    learning_rates = [1e-4]
    num_rnn_layers = [2]

    for rnn_layers in num_rnn_layers:
        for learning_rate in learning_rates:
            #while True:
            #    try:
            train_transducer(rnn_layers, learning_rate)
            #        break
            #    except Exception:
            #        print("caught nan")
"""


def test_transducer():
    dataset = SpectrogramDataset('data/CommonVoice/valid_train.h5')
    norm_transform = Normalize(dataset)
    decoder = RNNTransducer(dataset.char_to_ix)
    dataset.set_transform(norm_transform)
    batch_size = 4
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

    rnn_layers = 2

    model = TransducerModel(input_dim, hidden_dim, output_dim, rnn_layers, batch_size)
    model.to(device)
    checkpoint = torch.load("/home/grigorii/model_dicts/transducer_epoch_42.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    x, y = dataset.generate_test()
    # Get the distributions
    transcript_dist = model.infer(x)
    print(y)
    pred_rnn_cell = torch.nn.GRUCell(input_size = 29, hidden_size = hidden_dim)
    pred_rnn_cell.weight_ih = model.prediction_gru.weight_ih_l0
    pred_rnn_cell.weight_hh = model.prediction_gru.weight_hh_l0
    pred_rnn_cell.bias_ih = model.prediction_gru.bias_ih_l0
    pred_rnn_cell.bias_hh = model.prediction_gru.bias_hh_l0
    # First vector in the sequence needs to be zeros
    transcript_dist = transcript_dist.detach()
    beam_search_decoding(pred_rnn_cell, model.hidden_prediction[:, :1, :].contiguous().view(1, 256), transcript_dist, 100, model)

def beam_search_decoding(pred_rnn_cell, init_hidden, transcription_prob, beam_size, model):
    """
        Asymptotically correct decoding algorithm.
        transcription_prob      - T x K + 1 matrix, where transcription_prob[t, k] is the
                                  probability of k-th character of the alphabet at t-th timestep
        beam_size               - number of outputs to keep track of when expanding
                                  output_timeseries
    """
    alphabet =  {'a' : 0, 'b' : 1, 'c' : 2, 'd':  3,
                        'e': 4, 'f': 5, 'g': 6, 'h':7, 'i':8, 'j': 9, 'k': 10,
                        'l': 11, 'm' : 12, 'n' : 13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19,
                        'u' : 20, 'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24, 'z' : 25, "'" : 26, " ": 27, "_": 28}

    # Number of timesteps in the phoneme sequence50
    transcription_prob = transcription_prob[0]
    T = transcription_prob.shape[0]
    print("transcript prob shape", transcription_prob.shape)
    onehot_vecs = torch.eye(len(alphabet)).cuda()
    # Initialize with the first blank character
    blank_char_id = 0
    blank_in_tensor = torch.zeros([len(alphabet)]).cuda()
    blank_in_tensor.unsqueeze_(0)
    out, first_hid = run_gru_step(blank_in_tensor, init_hidden, pred_rnn_cell, model)
    # A and B are dictionaries with key as the output character sequence candidate,
    # and with value as the approximate probability of outputting this sequence,
    # the hidden state of the prediction RNN used to compute this sequence and the
    # last-step prediction rnn output for this sequence.
    A = {}
    B = {'': [0, first_hid, out]}

    for t in range(T):
        print(t)
        A = B
        B = {}
        # For each y, add probabilties of extending prefixes (among current beams)
        # of y to y during this timestep t
        for y in A:
            for yhat in A:
                if (is_prefix(yhat, y)):
                    A[y][0] = log_sum(A[y][0], prefix_ext_prob(yhat, A[yhat], y, transcription_prob[t], pred_rnn_cell, onehot_vecs, alphabet, model))
        # Find the most probable in A
        ymost = most_prob_y(A)
        # While B contains less than beam_size elements more
        # probable than the most probable in A

        while num_greater_than_ymost(ymost, A, B, beam_size):
            # Most probable in A
            prob_ymost, hid_ymost, out_ymost = A[ymost]
            del A[ymost]
            # Probability of extending ymost by the blank character
            last_output = combine_outputs(out_ymost, transcription_prob[t], model)
            ymost_next_bl = last_output[0][blank_char_id]
            prob_ymost_bl = prob_ymost + ymost_next_bl
            # Extending y_most by blank
            out_ymost_bl, ymost_hid_bl = run_gru_step(blank_in_tensor, hid_ymost, pred_rnn_cell, model)
            B[ymost] = [prob_ymost_bl, ymost_hid_bl, out_ymost_bl]
            # Probabilities of extending ymost by other elements of the alphabet, to
            # be used in the following loop
            # TODO: try using out_ymost before extending the sequence by blank,
            # because blank is for next timestep? Or no?
            #ymost_next_ch = combine_outputs(out_ymost_bl, transcription_prob[t], model)
            for k in alphabet.keys():
                char_id = alphabet[k]
#                prob_most_with_k = prob_ymost_bl + last_output[0][char_id + 1]
                prob_most_with_k = prob_ymost_bl + last_output[0][char_id + 1]

                ymost_with_k = ymost + k
                next_input_tensor = onehot_vecs[char_id]
                next_input_tensor.unsqueeze_(0)
                #out_ymost_k, ymost_hid_k = run_gru_step(next_input_tensor, ymost_hid_bl, pred_rnn_cell, model)
                out_ymost_k, ymost_hid_k = run_gru_step(next_input_tensor, hid_ymost, pred_rnn_cell, model)

                A[ymost_with_k] = [prob_most_with_k, ymost_hid_k, out_ymost_k]
            ymost = most_prob_y(A)
        # Remove all but beam_size most probable from B
        B = keep_beamsize_most_probable(B, beam_size)

    max = ""
    max_prob = -1e20
    print("SIZE OF B", len(B))
    print(B)
    for y in B:
        print(y)
        ylen = len(y)
        prob, _, _ = B[y]
        # FIX THAT
        prob /= ylen
        if (prob > max_prob):
            max_prob = prob
            max = y
    print("OUTPUTTING Y")
    print(max)
    print("OUTPUTTING Y")


def prefix_ext_prob(yhat, yhat_state, y, t_step_probs, pred_rnn_cell, onehot_vecs, alphabet, model):
    """
        P(y | yhat, t) - probability of extending yhat to y during timestep t
    """
    yhat_prob, yhat_hid_state, yhid_out = yhat_state

    # Start right after last character of yhat
    start_index = len(yhat)
    #print("SIZE OF ITERATION", len(y) - start_index)
    for u in range(start_index, len(y)):
        next_char_index = alphabet[y[u]]
        next_char_vector = onehot_vecs[next_char_index]
        # Get predictions for new character, replace the
        # old hidden state with the new one
        out, yhat_hid_state = run_gru_step(next_char_vector.unsqueeze_(0), yhat_hid_state, pred_rnn_cell, model)
        prob_vector = combine_outputs(out, t_step_probs, model)[0]
        #print("PREFIX EXTENSION PROB VALUE ", float(yhat_prob), float(prob_vector[alphabet[y[u]] + 1]))
        # Probability of outputting next character in this state
        yhat_prob = prob_vector[alphabet[y[u]] + 1] + yhat_prob
    #print("PREFIX EXTENSION", yhat_prob)
    return yhat_prob

def keep_beamsize_most_probable(B, beam_size):
    """
        Remove all but beam_size most probable sequences from B
    """
    if (beam_size >= len(B)):
        return B

    return {y: B[y] for y in sorted(B, key = lambda t: B[t][0], reverse=True)[:beam_size]}

def most_prob_y(A):
    """
        Find most probable sequence y in A so far
    """
    most_prob = -1e20
    best_y = None
    for y in A:
        prob, _, _ = A[y]
        prob = float(prob)
        if (prob > most_prob):
            best_y = y
            most_prob = prob
    return best_y

def num_greater_than_ymost(ymost, A, B, beam_size):
    """
        Determines if B contains less than beam_size elements more
        probable than the most probable (ymost) in A
    """
    ymost_prob, _, _ = A[ymost]
    count = 0
    #print(ymost_prob)
    for yb in B:
        yb_prob, _, _ = B[yb]
        #print(yb_prob)
        #print(yb_prob)
        if (yb_prob > ymost_prob):
            count += 1

    #print("=================================================")
    return count < beam_size

def run_gru_step(next_char_vector, hid_state, pred_rnn_cell, model):
    next_hid_state = pred_rnn_cell(next_char_vector, hid_state)
    output = model.hidden2density_pred(next_hid_state)
    return output, next_hid_state

def combine_outputs(out_ymost, transcription_prob, model):
    prob_sum = model.relu(out_ymost + transcription_prob)
    prob_unscaled = model.density2softmax(prob_sum)
    prob_scaled = F.log_softmax(prob_unscaled, 1)
    return prob_scaled

def is_prefix(yhat, y):
    """
        Determine if yhat is a proper prefix of y
    """
    if y.startswith(yhat) and len(y) > len(yhat):
        return True
    return False

def log_sum(a, b):
    """
        ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a))), unless one of them is zero
    """
    if (a != 0 and b != 0):
        summand = torch.log(1 + torch.exp(b - a))
        if (summand == float("inf")):
            return a
        return a + summand
    elif (a != 0):
        return a
    elif (b != 0):
        return b
    else:
        return 0


test_transducer()
