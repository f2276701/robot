#!/usr/bin/env python
#coding:utf-8

'''
Created on ($ date)
Update  on
Author:
Team:
Github: 
'''

import torch
import torch.nn as nn
import torch.nn.init as init


__all__ = ["RNN", "RNNPB", "LSTM", "LSTMPB", "GRU", "GRUPB"]


def init_weight(m):
    for name, param in m.named_parameters():
        print(name)
        if "bn" in name:
            continue
        if "weight" in name:
            init.orthogonal_(param)
        elif "bias" in name:
            init.normal_(param)


class RNN(nn.Module):
    """Implementation of recurrent neural network (RNN).
    """
    def __init__(self, args):
        """
        Args:
            closed_rate: float, indicating the interpolation rate of data and prediction
            closed_index: float, indicating the index range of applying closed loop. \
                          Usually we apply closed loop to the joint, and using open loop to the vision
        """
        super(RNN, self).__init__()
        self.unit_nums = args.unit_nums
        self.in_dims = args.in_dims
        self.out_dims = args.out_dims

        self.rnn = self._set_cell()
        self.fc1 = nn.Linear(in_features=self.in_dims,
                            out_features=self.unit_nums)
        self.act1 = torch.nn.functional.relu
        self.fc2 = nn.Linear(in_features=self.unit_nums,
                            out_features=self.out_dims)
        self.act2 = torch.tanh
        init_weight(self)

    def forward(self, inputs, state=None, *args, **kwargs): 
        output = self.fc1(inputs)
        output = self.act1(output)
        output, states = self.rnn(output)
        output = self.fc2(output)
        output = self.act2(output)
        if len(states) == 2:
            states = torch.cat(states, -1)
        return output, states

    def _set_cell(self):
        return nn.RNN(input_size=self.unit_nums,
                      hidden_size=self.unit_nums,
                      num_layers=1)


class RNNPB(RNN):
    def __init__(self, args, pb_unit):
        self.pb_dims = args.pb_dims
        super(RNNPB, self).__init__(args)
        self.pb_unit = pb_unit

    def _get_input(self, *args, **kwargs):
        cur_input = super(RNNPB, self)._get_input(*args, **kwargs)
        return torch.cat([cur_input, self.pb_unit], axis=-1)

    def _set_cell(self):
        return nn.RNNCell(input_size=self.in_dims + self.pb_dims,
                          hidden_size=self.unit_nums)


class GRU(RNN):
    def _set_cell(self):
        return nn.GRU(input_size=self.unit_nums,
                      hidden_size=self.unit_nums,
                      num_layers=1)


class GRUPB(RNNPB):
    def _set_cell(self):
        return nn.GRUCell(input_size=self.in_dims + self.pb_dims,
                          hidden_size=self.unit_nums)


class LSTM(RNN):
    def _set_cell(self):
        return nn.LSTM(input_size=self.unit_nums,
                       hidden_size=self.unit_nums,
                       num_layers=1)


class LSTMPB(RNNPB):
    def _set_cell(self):
        return nn.LSTMCell(input_size=self.in_dims + self.pb_dims,
                           hidden_size=self.unit_nums)



if __name__ == "__main__":
    from config import get_config
    rnn = RNNPB(get_config(), torch.ones(1,2))
    rnn.train()
    print([[n, i.requires_grad] for n,i in rnn.named_parameters()])
    rnn.eval()
    print([[n, i.requires_grad] for n,i in rnn.named_parameters()])
    #init_weight(rnn)

