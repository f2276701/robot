#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from utils import Logger, make_fig


class Model(object):
    def __init__(self, args):
        """Training model for RNN, RNNPB, LSTM, LSTMPB, GRU and GRUPB

        Attributes:
            name: str, network model name
            epoch: int, training epoch
            lr: float, learning rate
            closed_step: int, indicating staring closed loop in which step
            delay: int, indicating how far the rnn should predict
            model_save_path: save path for trained model
            model_save_iter: indicating the iteration of saving model
            log_path: save path for model output
            log_iter: indicating the iteration of saving model output
        """

        self.name = args.net
        self.epoch = args.epoch
        self.lr = args.lr
        self.closed_step = args.closed_step
        self.delay = args.delay

        self.model_save_path = args.model_save_path 
        self.model_save_iter = args.model_save_iter
        self.log_path = args.log_path
        self.log_iter = args.log_iter

        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        if args.tensorboard:
            self.writer = SummaryWriter(self.log_path + "tensorboard_log")

        self.loss_list = {"loss": .0}
        if args.validate:
            self.vali_loss_list = {"vali_loss": .0}

    def build(self, model, **kwargs):
        self.rnn = model(**kwargs).to(self.device)
        self.loss = nn.MSELoss()
        #self.loss = self._loss
        self.optim = optim.Adam(self.rnn.parameters(), lr=self.lr)
        #self.optim = optim.RMSprop(self.rnn.parameters(), lr=self.lr)
        #self.optim = optim.SGD(self.rnn.parameters(), lr=self.lr)
        self.logger = Logger(self)
        #self.scheduler = StepLR(self.optim, step_size=self.epoch // 2, gamma=0.1)
        #milestones = list(range(self.epoch // 5, self.epoch, self.epoch // 5))
        #self.scheduler = MultiStepLR(self.optim, milestones=milestones, gamma=0.5)
        self.scheduler = ExponentialLR(self.optim, gamma=0.999)
        print(self.rnn)

    def _loss(self, output, target):
        joint_loss = torch.pow(output[:, :, :8] - target[:, :, :8], 2).mean()
        vision_loss = torch.pow(output[:, :, 8:] - target[:, :, 8:], 2).mean()
        return joint_loss + vision_loss

    def load(self, pt_file):
        trained_model = torch.load(pt_file)
        self.rnn.load_state_dict(trained_model)

    def train(self, train_loader, vali_loader=None, vali_pb_list=None):
        closed_flag = False
        lr_reduce_epoch = 2000
        for epoch in range(1, self.epoch + 1):
            loss_sum = .0
            if epoch == self.closed_step and closed_flag is False:
                print("=" * 5, "Start closed loop", "=" * 5)
                closed_flag = True
            for i, (batch_x, batch_y) in enumerate(train_loader, 1):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optim.zero_grad()
                output_log, state_log = self.rnn(batch_x[:, :-self.delay, :], 
                                                 closed_flag=closed_flag)
                loss = self.loss(batch_y[:, self.delay:, :], output_log)
                loss_sum += loss.item()

                loss.backward()
                self.optim.step()

            if epoch % lr_reduce_epoch == 0:
                self.scheduler.step()

            self.loss_list["loss"] = loss_sum / i
            if epoch % self.log_iter == 0:
                if vali_loader is not None:
                    self.test(vali_loader, vali_pb_list)

                output = output_log.detach().cpu().numpy()
                np.savetxt("{}{}_result_{}".format(self.log_path, self.name, epoch), 
                           output[0], delimiter=",")
                data = batch_y.detach().cpu().numpy()
                self.result_img = make_fig([[output[0], data[0]]])

                self.logger(epoch)
                plt.close(self.result_img)

            if epoch % self.model_save_iter == 0:
                torch.save(self.rnn.state_dict(), 
                           "{}{}_{:0>6}.pt".format(self.model_save_path, 
                                                   self.name, epoch))

    def test(self, data_loader, pb_list=None):
        loss_sum = .0
        if pb_list is not None:
            prev_pb = self.rnn.pb_unit
            self.rnn.pb_unit = pb_list

        for i, (batch_x, batch_y) in enumerate(data_loader, 1):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            output_log, state_log = self.rnn(batch_x[:, :-self.delay, :], 
                                             closed_flag=False)
            loss = self.loss(batch_y[:, self.delay:, :], output_log)
            loss_sum += loss.item()

        if pb_list is not None:
            self.rnn.pb_unit = prev_pb

        if hasattr(self, "vali_loss_list"):
            self.vali_loss_list["vali_loss"] = loss_sum / i

        output_log = output_log.detach().cpu().numpy()
        state_log = state_log.detach().cpu().numpy()
        return output_log, state_log, loss_sum / i

    def predict(self, inputs, init_state, timesteps):
        output_log, state_log = [], []
        closed_idx = self.rnn.closed_index
        cur_input, state = inputs[:,0,:], init_state
        for i in range(1, timesteps + 1):
            output, state = self.rnn.step(cur_input, state=state)
            cur_input = torch.cat([output[:, :closed_idx], 
                                   inputs[:,i, closed_idx:]], dim=-1)

            output_log.append(output)
            state_log.append(torch.cat(state))
        output_log = torch.stack(output_log, dim=1).detach().cpu().numpy()
        state_log = torch.stack(state_log, dim=1).detach().cpu().numpy()
        return output_log, state_log


if __name__ == "__main__":

    import numpy as np
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from utils import sampling, DataGenerator

    args = parameters.get_config()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    #data = sampling(50, 100, "sin")
    #data2 = sampling(50, 100, "cos", phase=0.5*np.pi)
    #data = np.concatenate([data, data2], 0)
    data, noised_data = sampling(100, 100, "sin", noise=True)
    generator = DataGenerator(noised_data, data)
    inputs = DataLoader(generator, batch_size = args.batch_size) 
    trainer = Trainer(args)
    #trainer.build_model(RNN)
    trainer.build_model(LSTM)
    #trainer.train(inputs)

    #rnn = trainer.model
    #rnn.eval()
    #output_log = []
    #for i in range(99):
    #    if i == 0:
    #        #cur_input = torch.zeros([1,2])
    #        cur_input = torch.zeros([1, 2])
    #        cur_input.data.numpy()[0, 0] = -0.75
    #        state = None

    #    #output, state = rnn(cur_input, state=state, gen=True, pb_list=torch.zeros([1, 2]))
    #    output, state = rnn(cur_input, state=state, gen=True)
    #    output_log.append(output)
    #    cur_input = output

    #output_log = torch.stack(output_log, dim=1)
    test_data, test_noised_data = sampling(1, 100, "sin", add_noise=True)
    #output_log, state_log = trainer.generate(torch.from_numpy(test_data), mode="offline")
    output_log, state_log = trainer.generate(torch.from_numpy(test_data), mode="online")

