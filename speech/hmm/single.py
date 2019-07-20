from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import scipy
from glob import glob
import soundfile as sf
from os import path
import numpy as np
import pickle as pkl
np.random.seed(seed=273)

class GaussianHMM(object):
    """
    Gaussian Hidden Markov Model.
    """
    def __init__(self, n_states, n_dims):
        """
        Set up Gaussian HMM
        ------
        input:
        n_states: number of states in HMM (note: one of them will be a final state)
        n_dims: number of dimensions (13 MFCCs for this assignment)
        """
        self.n_states = n_states
        self.n_dims = n_dims

    def init_gaussian_params(self, X):
        """
        Initialize Gaussian parameters
        ------
        input:
        X: list of 2d-arrays with shapes (Ti, 13) for example i: each is a matrix of MFCCs for a digit utterance
        ------
        initialize mu and sigma for each state's Gaussian (where sigma is a diagonal covariance)
        """
        self.mu = np.zeros((self.n_states, self.n_dims))
        self.sigma = np.zeros((self.n_states, self.n_dims))

        i, lens = 0, [int(len(x) / self.n_states) for x in X]
        for s in range(self.n_states):
            X_section = []
            for x, l in zip(X, lens):
                X_section.append(x[i * l:(i + 1) * l])
            X_section_concat = np.concatenate(X_section)
            X_subset = X_section_concat[np.random.choice(len(X_section_concat), size=50, replace=True)]
            self.mu[s] = X_subset.mean(axis=0)
            self.sigma[s] = X_subset.var(axis=0)
            i += 1

    def init_hmm_params(self):
        """
        Initialize HMM parameters
        ------
        initialize pi (starting probability vector) and A (transition probabilities)
        """
        self.pi = np.zeros(self.n_states)
        self.pi[0] = 1.
        self.A = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states - 1):
            self.A[s, s:s + 2] = .5
        self.A[-1, -1] = 1.

    def get_emissions(self, x):
        """
        Compute Gaussian log-density at X for a diagonal model.
        ------
        get (continuous) emission probabilities from the multivariate normal
        """
        T, _ = x.shape
        log_B = np.zeros((self.n_states, T))
        for s in range(self.n_states):
            log_B[s] = multivariate_normal.logpdf(x, mean=self.mu[s], cov=np.diag(self.sigma[s]))
        return log_B

    def forward(self, log_pi, log_A, log_B):
        """
        Forward algorithm.
        ------
        input:
        log_pi: 1d-array of shape n_states: log of start probability vector
        log_A: 2d-array of shape (n_states, n_states): log of transition probability matrix
        log_B: 2d-array of shape (n_states, Tx): log of emision probabilities (Note: Tx depends on x)
        ------
        output:
        log_alpha: 2d-array of shape (n_states, Tx): log probability to state i at time t
        """
        _, T = log_B.shape
        log_alpha = np.zeros(log_B.shape)
        for t in range(T):
            if t == 0:
                log_alpha[:, t] = log_B[:, t] + log_pi
            else:
                log_alpha[:, t] = log_B[:, t] + logsumexp(log_A.T + log_alpha[:, t - 1], axis=1)
        return log_alpha

    def backward(self, log_A, log_B):
        """
        Backward algorithm.
        ------
        input:
        log_A: 2d-array of shape (n_states, n_states): log of transition probability matrix
        log_B: 2d-array of shape (n_states, Tx): log of emision probabilities (Note: Tx depends on x)
        ------
        output:
        log_beta: 2d-array of shape (n_states, Tx): log probability from state i at time t
        """
        _, T = log_B.shape
        log_beta = np.zeros(log_B.shape)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                log_beta[:, t] = 0.
            else:
                log_beta[:, t] = logsumexp(log_A + log_B[:, t + 1] + log_beta[:, t + 1], axis=1)
        return log_beta

    def viterbi(self, log_pi, log_A, log_B):
        """
        Use viterbi algorithm to find the best path and associated log probability.
        ------
        input:
        log_pi: 1d-array of shape n_states: log of start probability vector
        log_A: 2d-array of shape (n_states, n_states): log of transition probability matrix
        log_B: 2d-array of shape (n_states, Tx): log of emision probabilities (Note: Tx depends on x)
        ------
        output:
        q: 1d-array of length T: optimal state sequence for observed sequence
        log_prob: log probability of observed sequence
        """
        _, T = log_B.shape
        log_delta = np.zeros(log_B.shape)
        for t in range(T):
            if t == 0:
                log_delta[:, t] = log_B[:, t] + log_pi
            else:
                log_delta[:, t] = log_B[:, t] + np.max(log_A.T + log_delta[:, t - 1], axis=1)

        q = np.zeros(T, dtype=np.int32)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                q[t] = np.argmax(log_delta[:, t])
                log_prob = log_delta[q[t], t]
            else:
                q[t] = np.argmax(log_delta[:, t] + log_A[:, q[t + 1]])

        return q, log_prob

    def score(self, x):
        """
        Use forward-backward algorithm to
        compute log probability and posteriors.
        ------
        input:
        x :2d-array of shape (T, 13): MFCCs for a single example
        ------
        output:
        log_prob :scalar: log probability of observed sequence
        log_alpha :2d-array of shape (n_states, T): log prob of getting to state at time t from start
        log_beta :2d-array of shape (n_states, T): log prob of getting from state at time t to end
        gamma :2d-array of shape (n_states, T): state posterior probability
        xi :2d-array of shape (n_states, n_states): state transition probability matrix
        """
        T = len(x)

        log_pi = np.log(self.pi) # starting log probabilities
        log_A = np.log(self.A) # transition log probabilities
        log_B = self.get_emissions(x) # emission log probabilities

        log_alpha = self.forward(log_pi, log_A, log_B)
        log_beta = self.backward(log_A, log_B)

        log_prob = logsumexp(log_alpha[:, T - 1])

        gamma = np.exp(log_alpha + log_beta - log_prob)

        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            xi[t] = np.exp(log_alpha[:, t, np.newaxis] + log_A + log_B[:, t + 1] + log_beta[:, t + 1] - log_prob)
        xi = xi.sum(axis=0) # sum over time
        xi /= xi.sum(axis=1, keepdims=True).clip(1e-1) # normalize by state probabilities (sum transitions over j)

        return log_prob, log_alpha, log_beta, gamma, xi

    def train(self, X):
        """
        Estimate model parameters.
        ------
        input:
        X: list of 2d-arrays of shape (Tx, 13): list of single digit MFCC features
        ------
        update model parameters (A, mu, sigma)
        """
        stats = {
            "gamma": np.zeros((self.n_states, 1)),
            "A": np.zeros((self.n_states, self.n_states)),
            "X": np.zeros((self.n_states, self.n_dims)),
            "X**2": np.zeros((self.n_states, self.n_dims))
        }

        for x in X:
            log_prob, log_alpha, log_beta, gamma, xi = self.score(x)

            stats["gamma"] += gamma.sum(axis=1, keepdims=True)
            stats["A"] += xi
            stats["X"] += gamma.dot(x)
            stats["X**2"] += gamma.dot(x**2)

        stats["gamma"] += 1
        stats["A"][:-1,:-1] += np.diag(np.full(self.n_states - 1, 1))

        self.mu = stats["X"] / stats["gamma"]
        self.sigma = ((self.mu**2 + stats["X**2"] - 2 * self.mu * stats["X"] + self.mu**2 * stats["gamma"]) / stats["gamma"])
        self.sigma = self.sigma.clip(1e-1)

        self.A = np.where(np.bitwise_or(self.A == 0.0, self.A == 1.0), self.A, stats["A"]) # update transition probabilities
        self.A /= self.A.sum(axis=1, keepdims=True) # normalize transition probabilities


def main():
    dataset = np.load("../../../hw3/hw3-assign/data/mfccs.npz")
    Xtrain, Ytrain = dataset["Xtrain"], dataset["Ytrain"]
    Xtest, Ytest = dataset["Xtest"], dataset["Ytest"]

    # Expected error rate:
    # 15 states/15 iterations: 0.9714 forward, 0.9679 viterbi
    # 25 states/25 iterations: 0.9821 forward, 0.9821 viterbi
    # 50 states/50 iterations: 0.9804 forward, 0.9804 viterbi

    n_states = 15
    n_dims = 13
    n_iter = 15
    model = dict()

    digits = range(10)

    for digit in digits:
        print("Training HMM for digit %d" % digit)
        Xtrain_digit = [x for x, y in zip(Xtrain, Ytrain) if y == digit]
        model[digit] = GaussianHMM(n_states=n_states, n_dims=n_dims)
        model[digit].init_gaussian_params(Xtrain_digit)
        model[digit].init_hmm_params()

        for i in range(n_iter):
            print("Starting iteration %d..." % i)
            model[digit].train(Xtrain_digit)

    print("Testing HMM")
    forward_accuracy, viterbi_accuracy = np.zeros(10), np.zeros(10)
    forward_confusion, viterbi_confusion = np.zeros((10, 10)), np.zeros((10, 10))
    for x, y in zip(Xtest, Ytest):
        T = len(x)

        forward_scores, viterbi_scores = [], []
        for digit in digits:
            log_pi = np.log(model[digit].pi)
            log_A = np.log(model[digit].A)
            log_B = model[digit].get_emissions(x)

            log_alpha = model[digit].forward(log_pi, log_A, log_B)
            forward_log_prob = logsumexp(log_alpha[:, T - 1])
            _, viterbi_log_prob = model[digit].viterbi(log_pi, log_A, log_B)

            forward_scores.append(forward_log_prob)
            viterbi_scores.append(viterbi_log_prob)

        forward_top_digit, forward_top_log_prob = sorted(zip(digits, forward_scores), key=lambda x: -x[1])[0]
        viterbi_top_digit, viterbi_top_log_prob = sorted(zip(digits, viterbi_scores), key=lambda x: -x[1])[0]

        forward_confusion[y, forward_top_digit] += 1.
        viterbi_confusion[y, viterbi_top_digit] += 1.

    forward_accuracy = np.diag(forward_confusion) / forward_confusion.sum(axis=1)
    viterbi_accuracy = np.diag(viterbi_confusion) / viterbi_confusion.sum(axis=1)

    print("forward accuracy (%.4f)" % forward_accuracy.mean(), forward_accuracy)
    print("viterbi accuracy (%.4f)" % viterbi_accuracy.mean(), viterbi_accuracy)

    with open("single_digit_model.pkl", "wb") as f:
        pkl.dump(model, f)

if __name__ == "__main__":
    main()
