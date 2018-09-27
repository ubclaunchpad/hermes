import numpy as np
import math
import copy

class RNNTransducer():

    def __init__(self, alphabet):
        """
            Encoding each character used during decoding as an integer value
            in both ways (char->int and int->char)

            alphabet                - a dictionary with characters as keys and their
                                      ids as values (one to one correspondence)
        """
        self.alphabet = copy.deepcopy(alphabet)
        self.alphabet[''] = 29
        self.alphabet_reverse = {}
        for key, value in self.alphabet.items():
            self.alphabet_reverse[value] = key
        self.alphabet_size = len(self.alphabet)

    def eval_forward_prob(self, prob_density, label):
        """
            Finds the forward probability of outputting a particular output sequence, encoded by
            prediction prob, and returns it with the alphas.

            prob_density            - U x T x (K + 1) probability distribution tensor, output of
                                      build_prob_density()
            label                   - a string
        """
        T = prob_density.shape[0]
        U = prob_density.shape[1]
        blank_index = self.alphabet_size - 1
        # Initialization of prediction lattice
        pred_lattice = np.zeros(shape = (T,  U))
        pred_lattice[0, 0] = 0
        pred_lattice[1, 0] = prob_density[0, 0, blank_index]
        next_char_index = self.alphabet[label[0]]
        pred_lattice[0, 1] = prob_density[0, 0, next_char_index]
        # Initialize "borders" of the lattice
        for t in range(2, T):
            pred_lattice[t, 0] = log_prod(pred_lattice[t - 1, 0], prob_density[t - 1, 0, blank_index])
        for u in range(2, U):
            next_char_index = self.alphabet[label[u - 1]]
            pred_lattice[0, u] = log_prod(pred_lattice[0, u - 1], prob_density[0, u - 1, next_char_index])
        # Predict
        for t in range(1, T):
            for u in range(1, U):
                next_char_index = self.alphabet[label[u - 1]]
                blank_prob = log_prod(pred_lattice[t - 1, u], prob_density[t - 1, u, blank_index])
                emit_prob = log_prod(pred_lattice[t, u - 1], prob_density[t, u - 1, next_char_index])
                pred_lattice[t, u] = log_sum(emit_prob, blank_prob)
        return np.exp(log_prod(pred_lattice[T - 1, U - 1], prob_density[T - 1, U - 1, blank_index])), pred_lattice

    def eval_backward_prob(self, prob_density, label):
        """
            Finds the backward probability of outputting a particular output sequence, encoded by
            prediction prob, and returns it with the betas.

            prob_density            - U x T x (K + 1) probability distribution tensor, output of
                                      build_prob_density()
            label                   - a string
        """
        T = prob_density.shape[0]
        U = prob_density.shape[1]
        blank_index = self.alphabet_size - 1
        # Initialization of rediction lattice
        pred_lattice = np.zeros(shape = (T,  U))
        pred_lattice[T - 1, U - 1] = prob_density[T - 1, U - 1, blank_index]
        for t in reversed(range(0, T - 1)):
            pred_lattice[t, U - 1] = log_prod(pred_lattice[t + 1, U - 1], prob_density[t + 1, U - 1, blank_index])
        for u in reversed(range(0, U - 1)):
            next_char_index = self.alphabet[label[u]]
            pred_lattice[T - 1, u] = log_prod(pred_lattice[T - 1, u + 1], prob_density[T - 1, u + 1, next_char_index])
        for t in reversed(range(0, T - 1)):
            for u in reversed(range(0, U - 1)):
                next_char_index = self.alphabet[label[u]]
                blank_prob = log_prod(pred_lattice[t + 1, u], prob_density[t + 1, u, blank_index])
                emit_prob = log_prod(pred_lattice[t, u + 1], prob_density[t, u + 1, next_char_index])
                pred_lattice[t, u] = log_sum(blank_prob, emit_prob)
        return np.exp(pred_lattice[0, 0]), pred_lattice

    def build_prob_density(self, transcription_prob, prediction_prob):
        """
            Generates U x T x (K + 1) probability distribution tensor. Concretely,
            prob_density[u, t, k] is the probability of outputting kth element of
            alphabet (self.alphabet_reverse[k]) at t-th timestep of transcription
            sequence and u-th timestep of prediction sequence. For details, see
            the paper.

            transcription_prob      - T x K + 1 matrix, where transcription_prob[t, k] is the
                                      probability of k-th character of the alphabet at t-th timestep
            prediction_prob         - U x K + 1 matrix, where prediction_prob[u, k] is the
                                      probability of k-th character of the alphabet at u-th timestep
        """
        T = transcription_prob.shape[0]
        U = prediction_prob.shape[0]
        Kp1 = len(self.alphabet)
        # Create T x U x (K + 1) array where sum_probs[t, u] = transcription_prob[t] +
        # prediction_prob[u]
        sum_probs = np.add(prediction_prob, transcription_prob.reshape(T, 1, Kp1))
        sum_probs_exp_normalize = np.log(np.sum(np.exp(sum_probs), axis = 2))
        # TODO: Do it properly, just take normaml softamax over third dimention
        for i in range(T):
            normalizing_factor = np.array([sum_probs_exp_normalize[i]]).T
            sum_probs[i] = sum_probs[i] - normalizing_factor
        return sum_probs

    def beam_search_decoding(self, transcription_prob, beam_size):
        """
            Asymptotically correct decoding algorithm.
            transcription_prob      - T x K + 1 matrix, where transcription_prob[t, k] is the
                                      probability of k-th character of the alphabet at t-th timestep
            beam_size               - number of outputs to keep track of when expanding
                                      output_timeseries
        """
        B = {'': 1}
        T = transcription_prob.shape[0]
        for t in range(T):
            A = B
            B = {}
            for y in A:
                pass

    def compute_gradient(log_probs, alphas, betas, labels):
        """
            From Awni Hannun:
            https://github.com/awni/transducer/blob/master/ref_transduce.py
        """
        blank_index = self.alphabet_size - 1
        T, U, _ = log_probs.shape
        grads = np.full(log_probs.shape, -float("inf"))
        log_like = betas[0, 0]

        grads[T-1, U-1, blank_index] = alphas[T-1, U-1]

        grads[:T-1, :, blank_index] = alphas[:T-1, :] + betas[1:, :]
        for u, l in enumerate(labels):
            grads[:, u, l] = alphas[:, u] + betas[:, u+1]

        grads = grads + log_probs - log_like
        grads = np.exp(grads)

        grads = -grads
        return grads

def log_sum(a, b):
    """
        ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a))), unless one of them is zero
    """
    if (a != 0 and b != 0):
        return a + np.log(1 + np.exp(b - a))
    elif (a != 0):
        return a
    elif (b != 0):
        return b
    else:
        return 0

def log_prod(a, b):
    """
        ln(ab) = ln(a) + ln(b), unless one of them is zero
    """
    if (a != 0 and b != 0):
        return a + b
    else:
        return 0

def test_eval_forward():
    alphabet = {'h': 0, 'e' : 1, 'l' : 2, 'o' : 3, '' : 4}
    dec = RNNTransducer(alphabet)
    f = np.array([[0.5, 0.1, 0.2, 0.1, 0.1], [0.2, 0.4, 0.2, 0.1, 0.1], [0.1, 0.5, 0.1, 0.2, 0.1], [0.2, 0.5, 0.1, 0.1, 0.1]])
    g = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.2, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]])
    prob_density = dec.build_prob_density(f, g)
    pred_1, alphas = dec.eval_forward_prob(prob_density, "he")
    print(np.isclose(pred_1, 0.00335503 * 0.15232817, 1e-9))
    pred_2, betas = dec.eval_backward_prob(prob_density, "he")
    print(np.isclose(pred_2, 0.00335503 * 0.15232817, 1e-9))

test_eval_forward()
