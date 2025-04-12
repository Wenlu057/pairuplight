import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from agents.dru import  *

from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as Fnn
# tf.compat.v1.disable_eager_execution()

def ortho_weights(shape, scale=np.sqrt(2)):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

def model_initializer(module):
    """ Parameter initializer for Atari models

    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTMCell':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()

class ACPolicy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        super(ACPolicy,self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step


    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self):
        self.optimizer = optim.Adam(params=self.get_param())
        self.objective = PPOObjective()


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_n, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_neighbor = 128, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        # super(LstmACPolicy, self).__init__()
        self.train_mode = True
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_fc_neighbor = n_fc_neighbor
        self.n_w = n_w
        self.n_n = n_n
        self.fc0_pi = nn.Sequential(nn.Linear(n_s, n_fc_wave),nn.ReLU(inplace=True))
        self.fc1_pi = nn.Sequential(nn.Linear(n_w, n_fc_wait),nn.ReLU(inplace=True))
        self.fc0_v = nn.Sequential(nn.Linear(n_s, n_fc_wave), nn.ReLU(inplace=True))
        self.fc1_v = nn.Sequential(nn.Linear(n_w, n_fc_wait), nn.ReLU(inplace=True))
        self.fc2_v = nn.Sequential(nn.Linear(n_n, n_fc_neighbor), nn.ReLU(inplace=True))
        self.lstm_pi = nn.LSTMCell(input_size=(n_fc_wave), hidden_size=self.n_lstm)
        self.lstm_v = nn.LSTMCell(input_size=(n_fc_wave + n_fc_wait + n_fc_neighbor), hidden_size=self.n_lstm)
        self.message_mlp = nn.Sequential()
        self.message_mlp.add_module('linear1', nn.Linear(1, 128))
        self.message_mlp.add_module('relu1', nn.ReLU(inplace=True))

        self.pi_mlp = nn.Sequential()
        self.pi_mlp.add_module('linear1', nn.Linear(n_lstm, n_lstm))
        self.pi_mlp.add_module('relu1', nn.ReLU(inplace=True))
        self.pi_mlp.add_module('linear2',nn.Linear(n_lstm, n_a + 1))
        self.v_mlp = nn.Linear(n_lstm, 1)


        self.apply(model_initializer)
        self.pi_mlp.linear2.weight.data = ortho_weights(self.pi_mlp.linear2.weight.size(), scale=.01)
        self.v_mlp.weight.data = ortho_weights(self.v_mlp.weight.size())

        self.dru = DRU(2, True, False)

    def get_param(self):
        return list(self.parameters())

    def reset_parameters(self):
        self.fc0.linear0.reset_parameters()
        self.fc1.linear1.reset_parameters()
        self.lstm.reset_parameters()
        self.pi_mlp.linear2.reset_parameters()
        self.v_mlp.reset_parameters()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        if self.n_w == 0:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.hidden = torch.from_numpy(np.zeros((2, self.n_lstm * 2), dtype=np.float32))
        self.message = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))

        # self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        # self.reset_parameters()

    def forward(self, obs, hidden, message, out_type):
        wave = Variable(torch.from_numpy(obs[:, :self.n_s]).float())
        wait = Variable(torch.from_numpy(obs[:, self.n_s:(self.n_s + self.n_w)]).float())
        neighbor = Variable(torch.from_numpy(obs[:, (self.n_s + self.n_w):]).float())
        # hidden = self.hidden
        if 'p' in out_type:
            x0_p = self.fc0_pi(wave.view(-1, self.n_s))
            x1_p = self.fc1_pi(wait.view(-1, self.n_w))
            message_p = self.message_mlp(message.view(-1, 1))
            x_p = x0_p + x1_p + message_p

            h_pi, c_pi = self.lstm_pi(x_p, (hidden[:,0, :64], hidden[:,0, -64:]))
            pi_out = self.pi_mlp(h_pi)
            new_hidden_pi = torch.cat([h_pi, c_pi], 1)
            self.pi = pi_out[:,:4]


        if 'v' in out_type:
            x0_v = self.fc0_v(wave.view(-1, self.n_s))
            x1_v = self.fc1_v(wait.view(-1, self.n_w))
            x2_v = self.fc2_v(neighbor.view(-1, self.n_n))
            x = torch.cat([x0_v, x1_v, x2_v], 1)
            h_v, c_v = self.lstm_v(x, (hidden[:,1, :64], hidden[:,1, -64:]))
            v_out = self.v_mlp(h_v)
            new_hidden_v = torch.cat([h_v, c_v], 1)
            self.v = v_out

        if out_type is 'pv':
            if len(self.pi) == 1:
                new_hidden = torch.cat([new_hidden_pi, new_hidden_v], 0)
                hidden = new_hidden.detach()
                message = self.dru.forward(pi_out[:, 4:], train_mode=True)
                return self.pi, self.v, hidden, message.detach()
            else:
                return self.pi, self.v

        elif out_type is 'p':
            hidden = torch.from_numpy(np.zeros((2, self.n_lstm * 2), dtype=np.float32))
            hidden[0] = new_hidden_pi
            hidden = hidden.detach()
            message = self.dru.forward(pi_out[:, 4:], train_mode=False)
            return self.pi, hidden, message.detach()
        else:
            return self.v

        # return self.pi, self.v

    def backward(self,clip, pi, v, pi_old, v_old, action, advantage, returns, cur_lr, value_coef, entropy_coef, max_grad_norm=0.5):
        losses = self.objective(clip,
                                pi, v, pi_old, v_old,
                                action, advantage, returns)
        policy_loss, value_loss, entropy_loss = losses
        loss = policy_loss + value_loss * value_coef + entropy_loss * entropy_coef
        # print('policy loss: ' + str(policy_loss.item()))
        self.set_lr(self.optimizer, cur_lr)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        clip_grad_norm_(parameters=self.get_param(), max_norm=max_grad_norm)
        self.optimizer.step()

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs

