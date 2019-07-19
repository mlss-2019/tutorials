from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.autograd as autograd

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat

def end_pad_concat(labels):
    # Assumes last item in each example is the end token.
    batch_size = len(labels)
    end_tok = labels[0][-1]
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    return cat_labels

class LinearND(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        nn.Module.__init__(self)
        self.fc = nn.Linear(*args, **kwargs)

    def forward(self, x):
        """
        Forwards ND array (where the last dimension is treated as the feature dimension).
        """
        size = x.size()
        n = np.prod(size[:-1])
        out = x.contiguous().view((n, size[-1]))
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

class Seq2Seq(nn.Module):

    def __init__(self, feat_dim, vocab_size, attention, config):
        """
        Initialize Seq2Seq model instance.
        ----
        feat_dim: MFCC feature dimension (13)
        vocab_size: size of output vocabulary
        attention: instance of desired Attention mechanism
        config: dictionary defining the model configuration
        """
        nn.Module.__init__(self)

        self.enc_rnn = nn.LSTM(batch_first=True, input_size=feat_dim, **config["encoder"])

        self.embedding = nn.Embedding(vocab_size, config["decoder"]["input_size"])
        self.dec_rnn = nn.LSTM(batch_first=True, **config["decoder"])
        self.attend = attention

        self.fc = LinearND(self.enc_rnn.hidden_size, vocab_size - 1)  # predict vocab_size - 1 classes since we never predict start token

        self.sample_prob = config.get("sample_prob", 0)
        self.scheduled_sampling = self.sample_prob != 0

    @property
    def is_cuda(self):
        """
        returns whether or not the model is on the GPU
        """
        return self.parameters().__next__().is_cuda

    def collate(self, inputs, labels):
        """
        Preps input acoustic sequence and label sequence for being fed into the model.
        ----
        inputs: input MFCCs
        labels: digit labels
        ----
        Returns LongTensors or CudaTensors
        """
        inputs = zero_pad_concat(inputs)
        labels = end_pad_concat(labels)
        inputs = autograd.Variable(torch.from_numpy(inputs))
        labels = autograd.Variable(torch.from_numpy(labels))
        if self.is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        return inputs, labels

    def loss(self, x, y):
        """
        Computes the loss given the input x and targets y.
        x: (batch, T, feat_dim): Tensor
        y: (batch, L): Tensor
        ----
        Returns average loss over batch.
        """
        out, alis = self.forward_impl(x, y)

        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:, 1:].contiguous().view((-1))
        loss = nn.functional.cross_entropy(out, y, size_average=False)
        return loss / batch_size

    def forward_impl(self, x, y):
        """
        Forward inputs and targets and get out output sequnce prediction and attention alignments.
        x: (batch, T, feat_dim): Tensor
        y: (batch, L): Tensor
        ----
        Returns output predictions and attention alignment
        """
        x = self.encode(x)
        out, alis = self.decode(x, y)
        return out, alis

    def forward(self, x, y):
        """
        Forward inputs and targets and get out output sequnce prediction.
        ----
        x: (batch, T, feat_dim): Tensor
        y: (batch, L): Tensor
        ----
        Returns output predictions
        """
        return self.forward_impl(x, y)[0]

    def encode(self, x):
        """
        Encode input acoustic sequence x.
        x: (batch, T, feat_dim): Tensor
        ----
        Returns encoded hidden states
        """
        x, h = self.enc_rnn(x)
        return x

    def decode(self, x, y):
        """
        Decode sequence prediction from encoded hidden states (x).
        ------
        x: (batch, T, hidden_dim)
        y: (batch, L)
        ----
        Returns output predictions and attention alignments
        """

        inputs = self.embedding(y[:, :-1])

        out = []
        aligns = []
        ax, hx, sx = None, None, None
        for t in range(y.size()[1] - 1):
            sample = (out and self.scheduled_sampling)
            if sample and random.random() < self.sample_prob:
                ix = torch.max(out[-1], dim=2)[1]
                ix = self.embedding(ix)
            else:
                ix = inputs[:, t:t + 1, :]

            if sx is not None:
                ix = ix + sx

            ox, hx = self.dec_rnn(ix, hx=hx)
            sx, ax = self.attend(x, ox, ax)
            aligns.append(ax)
            out.append(self.fc(ox + sx))

        return torch.cat(out, dim=1), torch.stack(aligns, dim=1)

    def decode_step(self, x, y, state=None, softmax=False):
        """
        Decode step of sequence prediction from encoded hidden states (x).
        -----
        x: shape (batch, T, hidden state dimension) Tensor of encoder hidden states
        y: Tensor of predicted labels
        """
        if state is None:
            hx, ax, sx = None, None, None
        else:
            hx, ax, sx = state

        ix = self.embedding(y)
        if sx is not None:
            ix = ix + sx
        ox, hx = self.dec_rnn(ix, hx=hx)
        sx, ax = self.attend(x, ox, ax=ax)
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        if softmax:
            out = nn.functional.log_softmax(out, dim=1)
        return out, (hx, ax, sx)

    def infer_decode(self, x, y, end_tok, max_len):
        """
        Decode starting from start tok y.
        ----
        x: encoded hidden states
        y: start token
        end_tok: end token
        max_length: max length to decode
        ----
        Returns probabilities and argmax sequence predictions
        """
        probs = []
        argmaxs = [y]
        state = None
        for e in range(max_len):
            out, state = self.decode_step(x, y, state=state)
            probs.append(out)
            y = torch.max(out, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            if torch.sum(y.data == end_tok) == y.numel():
                break

        probs = torch.cat(probs)
        argmaxs = torch.cat(argmaxs, dim=1)
        return probs, argmaxs

    def infer(self, x, y, max_len=200):
        """
        Infer a likely output.
        ----
        x: MFCC inputs
        y: groundtruth inputs
        ----
        Returns sequence prediction
        """
        x = self.encode(x)

        start_tok = y.data[:, 0:1]  # needs to be the start token
        end_tok = y.data[0, -1]

        _, argmaxs = self.infer_decode(x, start_tok, end_tok, max_len)
        argmaxs = argmaxs.cpu().data.numpy()
        return [seq.tolist() for seq in argmaxs]
