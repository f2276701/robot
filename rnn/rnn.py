#!/usr/bin/env python
#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.init as init


__all__ = ["RNN", "RNNPB", "LSTM", "LSTMPB", "GRU", "GRUPB"]

def init_weight(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            init.orthogonal_(param)
        elif "bias" in name:
            init.normal_(param)
            if isinstance(m, (LSTM, LSTMPB)) and ("bias_ih" in name or "bias_hh" in name):
                ### init forgetbias to 1 to remember more information
                param.data[m.unit_nums:2*m.unit_nums].fill_(1)


class RNN(nn.Module):
    """Implementation of recurrent neural network (RNN).
    """
    def __init__(self, args):
        """
        Args:
            unit_nums: int, hidden state dim
            in_dims: int, input dim
            out_dims: int, output dim
            closed_rate: float, indicating the interpolation rate of data and prediction
            closed_index: float, indicating the index range of applying closed loop. \
                          Usually we apply closed loop to the joint, and using open loop to the vision
        """
        super(RNN, self).__init__()
        self.unit_nums = args.unit_nums
        self.in_dims = args.in_dims
        self.out_dims = args.out_dims

        self.closed_rate = args.closed_rate
        self.closed_index = args.closed_index

        self.init_h = nn.Parameter(torch.randn(1, self.unit_nums), 
                                        requires_grad=True)

        self.rnn = self._set_cell()
        self.fc1 = nn.Linear(in_features=self.unit_nums,
                            out_features=self.unit_nums)
        self.act1 = torch.tanh
        self.fc2 = nn.Linear(in_features=self.unit_nums,
                            out_features=self.out_dims)
        self.act2 = torch.tanh
        init_weight(self)

    def steps(self, inputs, state=None, closed_flag=False):
        state_log = []
        output_log = []
        prev_output = None

        inputs = torch.unbind(inputs, 1)
        for step, cur_input in enumerate(inputs, 1):
            output, state = self.step(cur_input, state, prev_output)
            if isinstance(self.rnn, nn.LSTMCell):
                state_log.append(torch.cat(state, dim=-1))
            else:
                state_log.append(state)
            output_log.append(output)

            # if closed_flag is True, model would take output of previous step as input of current step
            if closed_flag:
                prev_output = output

        state_log = torch.stack(state_log, dim=1)
        output_log = torch.stack(output_log, dim=1)
        return output_log, state_log

    def step(self, cur_input, state=None, prev_output=None):
        cur_input = self._get_input(cur_input, prev_output)
        if state is None:
            if isinstance(self.rnn, nn.LSTMCell):
                state = [self.init_h.repeat(cur_input.shape[0], 1),
                         self.init_c.repeat(cur_input.shape[0], 1)]
            else:
                state = self.init_h.repeat(cur_input.shape[0], 1)

        state = self.rnn(cur_input, state)
        if isinstance(self.rnn, nn.LSTMCell):
            output = self.act1(self.fc1(state[0]))
        else:
            output = self.act1(self.fc1(state))
        output = self.act2(self.fc2(output))
        return output, state

    def forward(self, inputs, state=None, *args, **kwargs):
        return self.steps(inputs, state, *args, **kwargs)

    def _get_input(self, cur_input, prev_output): 
        # mix the open and closed loop to imporve the traning stability
        if prev_output is not None:
            inter_input = (1.0 - self.closed_rate) * cur_input + self.closed_rate * prev_output
            cur_input = torch.cat([inter_input[:, :self.closed_index],
                                   cur_input[:, self.closed_index:]], -1)
        return cur_input

    def _set_cell(self):
        #overload this method for different rnn cell
        return nn.RNNCell(input_size=self.in_dims,
                          hidden_size=self.unit_nums)


class RNNPB(RNN):
    """Implementation of recurrent neural network with Parametric Bias (RNNPB).
    """
    def __init__(self, args, pb_unit):
        """
        Args:
            pb_dims: int, pb state dim
            pb_unit: tensor, pb tensor
        """
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
        return nn.GRUCell(input_size=self.in_dims,
                          hidden_size=self.unit_nums)


class GRUPB(RNNPB):
    def _set_cell(self):
        return nn.GRUCell(input_size=self.in_dims + self.pb_dims,
                          hidden_size=self.unit_nums)


class LSTM(RNN):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.init_c = nn.Parameter(torch.randn(1, self.unit_nums), 
                                   requires_grad=True)
    def _set_cell(self):
        return nn.LSTMCell(input_size=self.in_dims,
                           hidden_size=self.unit_nums)


class LSTMPB(RNNPB):
    def __init__(self, *args, **kwargs):
        super(LSTMPB, self).__init__(*args, **kwargs)
        self.init_c = nn.Parameter(torch.randn(1, self.unit_nums), 
                                   requires_grad=True)
    def _set_cell(self):
        return nn.LSTMCell(input_size=self.in_dims + self.pb_dims,
                           hidden_size=self.unit_nums)



if __name__ == "__main__":
    from config import get_config
    rnn = LSTM(get_config())
    #rnn = LSTMPB(get_config(), torch.ones(1,2))
    #init_weight(rnn)

