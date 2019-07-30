import json
import numpy as np
import random
import soundfile as sf
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud

from torch.utils.data import DataLoader

from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct
from scipy.signal import stft

from collections import defaultdict
from copy import deepcopy
from glob import glob

def pre_emphasis(x):
    """
    Applies pre-emphasis step to the signal.
    - balance frequencies in spectrum by increasing amplitude of high frequency 
    bands and decreasing the amplitudes of lower bands
    - largely unnecessary in modern feature extraction pipelines
    ------
    :in:
    x, array of samples
    ------
    :out:
    y, array of samples
    """
    y = np.append(x[0], x[1:] - 0.97 * x[:-1])

    return y

def hamming(n):
    """
    Hamming method for weighting components of window.
    Feel free to implement more window functions.
    ------
    :in:
    n, window size
    ------
    :out:
    win, array of weights to apply along window
    """
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

    return win

def windowing(x, size, step):
    """
    Window and stack signal into overlapping frames.
    ------
    :in:
    x, array of samples
    size, window size in number of samples (Note: this may need to be a power of 2)
    step, window shift in number of samples
    ------
    :out:
    frames, 2d-array of frames with shape (number of windows, window size)
    """
    xpad = np.append(x, np.zeros((size - len(x) % size)))

    T = (len(xpad) - size) // step
    frames = np.stack([xpad[t * step:t * step + size] for t in range(T)])

    return frames

def discrete_fourier_transform(x):
    """
    Compute the discrete fourier transform for each frame of windowed signal x.
    Typically, we talk about performing the DFT on short-time windows
    (often referred to as the Short-Time Fourier Transform). Here, the input
    is a 2d-array with shape (window size,  number of windows). We want to
    perform the DFT on each of these windows.
    Note: this can be done in a vectorized form or in a loop.
    --------
    :in:
    x, 2d-array of frames with shape (window size, number of windows)
    --------
    :out:
    X, 2d-array of complex spectrum after DFT applied to each window of x
    """

    n = len(x)
    indices = np.arange(n)
    M = np.exp(-2j * np.pi * np.outer(indices, indices) / n)
    return np.dot(M, x)

def fast_fourier_transform(x):
    """
    Fast-fourier transform. Effiicient algorithm for computing the DFT.
    --------
    :in:
    x, 2d-array of frames with shape (window size, number of windows)
    --------
    :out:
    X, 2d-array of complex spectrum after DFT applied to each window of x
    """
    fft_size = len(x)

    if fft_size <= 16:
        X = discrete_fourier_transform(x)

    else:
        indices = np.arange(fft_size)
        even = fast_fourier_transform(x[::2])
        odd = fast_fourier_transform(x[1::2])
        m = np.exp(-2j * np.pi * indices / fft_size).reshape(-1, 1)
        X = np.concatenate([even + m[:fft_size // 2] * odd, even + m[fft_size // 2:] * odd])

    return X

def mel_filterbank(nfilters, fft_size, sample_rate):
    """
    Mel-warping filterbank.
    You do not need to edit this code; it is needed to contruct the mel filterbank
    which we will use to extract features.
    --------
    :in:
    nfilters, number of filters
    fft_size, window size over which fft is performed
    sample_rate, sampling rate of signal
    --------
    :out:
    mel_filter, 2d-array of (fft_size / 2, nfilters) used to get mel features
    mel_inv_filter, 2d-array of (nfilters, fft_size / 2) used to invert
    melpoints, 1d-array of frequencies converted to mel-scale
    """
    freq2mel = lambda f: 2595. * np.log10(1 + f / 700.)
    mel2freq = lambda m: 700. * (10**(m / 2595.) - 1)

    lowfreq = 0
    highfreq = sample_rate // 2

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)

    melpoints = np.linspace(lowmel, highmel, 1 + nfilters + 1)

    # must convert from freq to fft bin number
    fft_bins = ((fft_size + 1) * mel2freq(melpoints) // sample_rate).astype(np.int32)

    filterbank = np.zeros((nfilters, fft_size // 2))
    for j in range(nfilters):
        for i in range(fft_bins[j], fft_bins[j + 1]):
            filterbank[j, i] = (i - fft_bins[j]) / (fft_bins[j + 1] - fft_bins[j])
        for i in range(fft_bins[j + 1], fft_bins[j + 2]):
            filterbank[j, i] = (fft_bins[j + 2] - i) / (fft_bins[j + 2] - fft_bins[j + 1])

    return filterbank.T / filterbank.sum(axis=1).clip(1e-16)

def extract_mfcc_features(wav):
    """
    MFCC feature extraction.
    This function is concise representation of the process you started with above.
    This function will be used to extract features from tidgits examples (Downloaded from canvas)
    to perform a simple single digit recognition task.
    --------
    :in:
    signal : array of audio samples
    rate   : sampling rate
    --------
    :return: normalized mfcc features (number of frames, number of cepstral coefficients)
    """
    signal, rate = sf.read(wav)

    size = 128
    step = size // 2
    nfilters = 26
    ncoeffs = 13

    # pre-emphasize signal
    pre_emphasized_signal = pre_emphasis(signal)

    # window signal
    frames = windowing(pre_emphasized_signal, size, step) * hamming(size)

    # compute complex spectrum (Note: this produces symmetric output, only need first half)
    spectrum = fast_fourier_transform(frames.T).T
    spectrum = spectrum[:, :size // 2]

    # compute spectrum magnitude (typically what is meant by spectrogram)
    magnitude = np.abs(spectrum)

    # get spectrum power
    power = magnitude**2 / size

    # apply mel warping filters to power spectrum and take log10
    mel_filter = mel_filterbank(nfilters=nfilters, fft_size=size, sample_rate=rate)
    log_mel_fbank = np.log10(power.dot(mel_filter).clip(1e-16))

    # compute MFCCs using discrete cosine transform
    mfccs = dct(log_mel_fbank, type=2, axis=1, norm='ortho')

    # keep only first 'ncoeffs' cepstral coefficients
    mfccs = mfccs[:,:ncoeffs]

    return mfccs


def compute_mean_std(wavfiles):
    samples = np.vstack([extract_mfcc_features(wav) for wav in wavfiles])
    return samples.mean(axis=0), samples.std(axis=0)

class Preprocessor():

    START = "<s>"
    END = "</s>"

    def __init__(self, data_json, max_samples=100, start_and_end=True):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
        """
        data = read_data_json(data_json)

        # Compute data mean, std from sample
        wavfiles = [d['audio'] for d in data]
        random.shuffle(wavfiles)
        self.mean, self.std = compute_mean_std(wavfiles[:max_samples])
        self._input_dim = self.mean.shape[0]

        # Make char map
        chars = list(set(t for d in data for t in d['text']))
        if start_and_end:
            # START must be last to be easily excluded in output classes
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v: k for k, v in self.int_to_char.items()}

    def encode(self, text):
        text = list(text)
        if self.start_and_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]

    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        if not self.start_and_end:
            return text

        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]

    def preprocess(self, wav, text):
        inputs = extract_mfcc_features(wav)
        inputs = (inputs - self.mean) / self.std
        targets = self.encode(text)
        return inputs, targets

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)


class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):
        self.data = [preproc.preprocess(d["audio"], d["text"]) for d in read_data_json(data_json)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)

def make_loader(dataset_json, preproc, batch_size):
    dataset = AudioDataset(dataset_json, preproc, batch_size)
    sampler = BatchRandomSampler(dataset, batch_size)
    collate_fn = lambda batch: zip(*batch)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]
