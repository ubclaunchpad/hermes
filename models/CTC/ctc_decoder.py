import numpy as np
import math

class CTCDecoder():

    def __init__(self, alphabet):
        """
            Encoding each character used during decoding as an integer value
            in both ways (char->int and int->char)

            alphabet        - a dictionary with characters as keys and their
                              ids as values (one to one correspondence)
        """
        self.alphabet = alphabet
        self.alphabet[''] = 0
        self.alphabet_reverse = {}
        for key, value in self.alphabet.items():
            self.alphabet_reverse[value] = key

    def eval_forward_prob(self, output_timeseries, label):
        """ Finds the CTC score for the string label given the RNN output distributions
            for all timesteps.

            output_timeseries       - T x D numpy array, where T
                                      is the length of timeseries of character distributions,
                                      and D is the size of the alphabet with the blank character
            label                   - a string
        """
        T = output_timeseries.shape[0]
        aug_label = self.preprocess_label(label)
        L = len(aug_label)
        # Converting to logprobs, so that we don't underflow
        output_timeseries = np.log(output_timeseries)
        # Initial probabilities
        # notation from the paper: alpha_t(s) = alpha[t, s]
        alpha_matrix = np.zeros(shape = (T, L))
        alpha_matrix[0, 0] = output_timeseries[0, aug_label[0]]
        alpha_matrix[0, 1] = output_timeseries[0, aug_label[1]]
        # ....
        # for all s > 1, alpha_matrix[0, s] = 0
        for t, char_dist in enumerate(output_timeseries[1:], 1):
            s = 0
            while (s < L):
                # probability that current character was already reached in previous
                # timesteps
                reached = alpha_matrix[t - 1, s]
                # probability of transitioning from previous character (blank or same character)
                # to the current
                prev_blank_same = alpha_matrix[t - 1, s - 1] if s >= 1 else 0
                alpha_hat = log_of_sum(reached, prev_blank_same)
                # adding probability of transitioning from previous distinct non-blank
                # character to current one
                prev_distinct = alpha_matrix[t - 1, s - 2] if s >= 2 else 0
                #  (repeated characters => need blank) or (current character is blank)
                if (aug_label[s - 2] == aug_label[s] or aug_label[s] == self.alphabet['']):
                    alpha_matrix[t, s] = prod_of_logs(output_timeseries[t, aug_label[s]], alpha_hat)
                # previous character is a blank between two unique characters
                else:
                    alpha_matrix[t, s] = prod_of_logs(output_timeseries[t, aug_label[s]],
                                                      log_of_sum(alpha_hat, prev_distinct))
                s += 1
            # normalize the alphas for current timestep so that we don't underflow
        return np.exp(alpha_matrix[T - 1, L - 1]) + np.exp(alpha_matrix[T - 1, L - 2])

    def best_path_decoding(self, output_timeseries):
        """
            The most elementary decoding scheme - take character predictions with
            highest probabilities at each timestep. Ignores that there can be
            multiple paths with higher summed probability corresponding to a
            different label.
        """
        best_path_indices = np.argmax(output_timeseries, axis = 1)
        best_path = [self.alphabet_reverse[i] for i in best_path_indices]
        return alignment_postprocess(best_path)

    def beam_search_decoding(self, output_timeseries, beam_size):
        """
            Asymptotically correct decoding algorithm
            output_timeseries       - T x D numpy array, where T
                                      is the length of timeseries of character distributions,
                                      and D is the size of the alphabet with the blank character
            beam_size               - number of outputs to keep track of when expanding
                                      output_timeseries
        """
        # Initialization
        #output_timeseries = np.log(output_timeseries)
        first_timestep = output_timeseries[0]
        if (beam_size > len(self.alphabet)):
            # All paths can fit in the set of tracked beams
            curr_beams = [(self.alphabet_reverse[i], (first_timestep[i], 0)) for i in range(len(self.alphabet))]
        else:
            best_alignments_ind = np.argpartition(first_timestep, -beam_size)[-beam_size:]
            # Expand over beam_size most probable first characters, keeping the probability of
            # this character
            curr_beams = [(self.alphabet_reverse[i], (first_timestep[i], 0)) for i in best_alignments_ind]
        # Step
        for i, char_dist in enumerate(output_timeseries[1:], 1):
            outputs_dict = {}
            # Create dictionary where keys are new outputs and values are
            # their probabilities
            for beam in curr_beams:
                output = beam[0]
                for new_char, j in self.alphabet.items():
                    expansion_prob = prod_of_logs(char_dist[j], log_of_sum(beam[1][0], beam[1][1]))
                    next_path = output + new_char
                    # Ok, now it can be a repeat or blank or distinct, 3 cases
                    # If blank,
                    if new_char == '':
                        if outputs_dict.get(output) is None:
                            outputs_dict[output] = (0, expansion_prob)
                        else:
                            outputs_dict[output] = (outputs_dict[output][0],
                                                    log_of_sum(outputs_dict[output][1], expansion_prob))
                    # If repeat, we should keep two versions:
                    # 1. the character added to the path is collapsed
                    # 2. the character added to the path is not collapsed (so we have
                    #    repeated characters in output, like 'll' in hello)
                    elif len(output) > 0 and new_char == output[-1]:
                        if outputs_dict.get(output) is None:
                            outputs_dict[output] = (prod_of_logs(char_dist[j], beam[1][0]), 0)
                        else:
                            outputs_dict[output] = (log_of_sum(prod_of_logs(char_dist[j], beam[1][0]),
                                                    outputs_dict[output][0]), outputs_dict[output][1])
                        if outputs_dict.get(next_path) is None:
                            outputs_dict[next_path] = (prod_of_logs(char_dist[j], beam[1][1]), 0)
                        else:
                            outputs_dict[next_path] = (log_of_sum(prod_of_logs(char_dist[j], beam[1][1]),
                                                        outputs_dict[next_path][0]), outputs_dict[next_path][1])
                    # If distinct, we just add this character
                    else:
                        if outputs_dict.get(next_path) is None:
                            outputs_dict[next_path] = (expansion_prob, 0)
                        else:
                            outputs_dict[next_path] = (log_of_sum(expansion_prob,
                                                       outputs_dict[next_path][0]), outputs_dict[next_path][1])
            curr_beams = sorted(outputs_dict.items(), key = lambda t: sum_for_max(t[1][0], t[1][1]),
                                                        reverse = True)[:beam_size]
            #endloop
        return (curr_beams[0][0], np.exp(curr_beams[0][1][0]) + np.exp(curr_beams[0][1][1]))

    def preprocess_label(self, label):
        """ Converts the labels to a sequence of character codes with
            a blank character between the original word's characters

            label                   - a string
        """
        aug_label = []
        aug_label.append(self.alphabet[''])
        for char in label:
            aug_label.append(self.alphabet[char])
            aug_label.append(self.alphabet[''])
        return aug_label

def log_of_sum(a, b):
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

def sum_for_max(a, b):
    if (a == 0 and b == 0):
        return float("-inf")
    else:
        return log_of_sum(a, b)

def prod_of_logs(a, b):
    """
        ln(ab) = ln(a) + ln(b), unless one of them is zero
    """
    if (a != 0 and b != 0):
        return a + b
    else:
        return 0

def alignment_postprocess(alignment):
    """
        Removes repeated adjacent characters. No need to remove blanks since
        they are represented by '' here.
    """
    label = []
    label.append(alignment[0])
    for i in range(len(alignment), 1):
        if (alignment[i] != alignment[i - 1]):
            label.append(alignment[i])
    return "".join(label)


def test_eval_forward():
    alphabet1 = {'c': 0, 'a' : 1, 't' : 2, 'd' : 3, 'o':  4, 'g': 5, '': 6}
    alphabet2 = {'h': 0, 'e' : 1, 'l' : 2, 'o' : 3, '' : 4}
    dec1 = CTCDecoder(alphabet1)
    dec2 = CTCDecoder(alphabet2)
    # Not valid distributions, but easy to compute
    output_timeseries_1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    # A bit more realistic
    output_timeseries_2 = np.array([[0.3, 0.2, 0.1, 0.3, 0.1], [0.1, 0.5, 0.1, 0.2, 0.1],
                                    [0.2, 0.2, 0.2, 0.2, 0.2], [0.6, 0.1, 0.1, 0.1, 0.1],
                                    [0.2, 0.2, 0.1, 0.3, 0.2], [0.1, 0.1, 0.1, 0.3, 0.1],
                                    [0.1, 0.1, 0.1, 0.3, 0.1]])
    print(np.isclose(dec1.eval_forward_prob(output_timeseries_1, "cat"), 0.0007, 1e-9))
    print(np.isclose(dec1.eval_forward_prob(output_timeseries_1, "dog"), 0.0007, 1e-9))
    print(np.isclose(dec2.eval_forward_prob(output_timeseries_2, "hello"), 0.0001344, 1e-9))

def test_beam_decoding():
    alphabet1 = {'c': 0, 'a' : 1, 't' : 2, 'd' : 3, 'o':  4, 'g': 5, '': 6}
    alphabet2 = {'h': 0, 'e' : 1, 'l' : 2, 'o' : 3, '' : 4}
    # Not valid distributions, but easy to compute
    output_timeseries_1 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    # A bit more realistic
    output_timeseries_2 = np.array([[0.3, 0.2, 0.1, 0.3, 0.1], [0.1, 0.5, 0.1, 0.2, 0.1],
                                    [0.2, 0.2, 0.2, 0.2, 0.2], [0.6, 0.1, 0.1, 0.1, 0.1],
                                    [0.2, 0.2, 0.1, 0.3, 0.2], [0.1, 0.1, 0.1, 0.3, 0.1],
                                    [0.1, 0.1, 0.1, 0.3, 0.1]])
    dec1 = CTCDecoder(alphabet1)
    dec2 = CTCDecoder(alphabet2)
    res_tuple = dec2.beam_search_decoding(output_timeseries_2, 200)
    print(np.isclose(dec2.eval_forward_prob(output_timeseries_2, res_tuple[0]), res_tuple[1], 1e-9))
