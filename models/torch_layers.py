# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    """
    Self attention layer: aggreagating a sequence into a single vector.
    This implementation uses the attention formula proposed by  Sukhbaatar etal. 2015
    https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf

    Usage:
    seq_len=10; bsz=16; in_dim=128
    attn = SelfAtnn(in_dim)
    x = torch.rand(seq_len, bsz, in_dim)  # 10x16x128
    y, a = attn(x)  # output y 16x128, attention weight a 10x16
    """
    def __init__(self, d_input, units=None):
        """
        :param d_input: input feature dimension
        :param units: dimension of internal projection, if None it will be set to d_input
        """
        super(SelfAttn, self).__init__()
        self.d_input = d_input
        self.units = units if units else d_input
        self.projection = nn.Linear(self.d_input, self.units)
        self.V = nn.Parameter(torch.Tensor(self.units, 1))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        # self.projection.weight.data.normal_()
        self.projection.weight.data.uniform_(-initrange, initrange)
        self.V.data.uniform_(-initrange, initrange)

    def forward(self, x, mask=None):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [seq_len, bsz, feat_dim]
        :return:  output tensor [bsz, feat_dim]
        """
        # ui = tanh(xW+b)
        ui = torch.tanh(self.projection(x))  # [seq_len, bsz, units]
        # a = softmax(uV)
        ai = F.softmax(torch.matmul(ui, self.V), dim=0)  # [seq_len, bsz, 1]
        if mask is not None:  # apply mask
            ai = ai * mask.unsqueeze(-1)  # [seq_len, bsz, 1]
            ai = ai / ai.sum(dim=0, keepdim=True)
        o = torch.sum(x * ai, dim=0)
        return o, ai.squeeze(-1)

    def extra_repr(self):
        return 'Sx?x%d -> ?x%d' % (self.d_input, self.d_input)


class ObjectClassifier(nn.Module):
    """
    perform log likelihood over sequence data ie. log(softmax), permute dimension
      accordingly to meet NLLLoss requirement
    Input: [seq_len, bsz, d_input]
    Output: [bsz, num_classes, seq_len]

    Usage:
    bsz=5; seq=16; d_input=1024; num_classes=10
    classiifer = ObjectClassifier(d_input, num_classes)
    x = torch.rand(seq, bsz, d_input)  # 16x5x1024
    out = classifier(x)  # 5x10x16
    """
    def __init__(self, d_input, num_classes):
        super(ObjectClassifier, self).__init__()
        self.d_input = d_input
        self.num_classes = num_classes
        self.linear = nn.Linear(d_input, num_classes)
        self.classifier = nn.LogSoftmax(dim=1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: (S,N,d_input)
        out = self.linear(x)  # (S,N,C), C = num_classes
        out = out.permute(1, 2, 0)  # (N,C,S)
        return self.classifier(out)  # (N,C,S)

    def extra_repr(self) -> str:
        return 'SxBx%d -> Bx%dxS' % (self.d_input, self.num_classes)


class FCSeries(nn.Module):
    """
    a set of FC layers separated by ReLU
    """
    def __init__(self, d_input, layer_dims=[], dropout=0.0, relu_last=True):
        super(FCSeries, self).__init__()
        self.nlayers = len(layer_dims)
        self.all_dims = [d_input] + layer_dims
        self.dropout_layer= nn.Dropout(p=dropout)
        self.relu_last = relu_last
        self.fc_layers = nn.ModuleList()
        for i in range(self.nlayers):
            self.fc_layers.append(nn.Linear(self.all_dims[i], self.all_dims[i+1]))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for i in range(self.nlayers):
            self.fc_layers[i].bias.data.zero_()
            self.fc_layers[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = x
        for i in range(self.nlayers):
            out = self.fc_layers[i](out)
            if i < self.nlayers-1:
                out = self.dropout_layer(F.relu(out))
            elif self.relu_last:  # last layer and relu_last=True
                out = F.relu(out)
        return out 

    def extra_repr(self):
        out = '?x%d' % self.all_dims[0]
        if self.nlayers == 0:
            out += ' -> (identity) %s' % out
        else:
            for i in range(self.nlayers):
                out += ' -> ?x%d' % self.all_dims[i+1]
        return out


class GreedyHashLoss(torch.nn.Module):
    def __init__(self):
        super(GreedyHashLoss, self).__init__()
        # self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])
        # self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])

    def forward(self, u):
        b = GreedyHashLoss.Hash.apply(u)
        # # one-hot to label
        # y = onehot_y.argmax(axis=1)
        # y_pre = self.fc(b)
        # loss1 = self.criterion(y_pre, y)
        loss = (u.abs() - 1).pow(3).abs().mean()
        return b, loss

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


