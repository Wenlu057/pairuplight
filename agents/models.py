"""
PairupLight model
@author: Wenlu Du
"""

import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import torch
from torch import nn
import copy
from agents.agent import *


class A2C():
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        super(A2C, self).__init__()
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')


    def _init_policy(self, n_s, n_a, n_w, n_n, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        n_neighbor = model_config.getint('num_neighbor')
        policy = LstmACPolicy(n_s, n_a, n_w, n_n, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft,n_fc_neighbor=n_neighbor, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        clip_init = model_config.get('clip_init')
        clip_decay = model_config.get('clip_decay')
        self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        self.clip_scheduler = Scheduler(clip_init, decay=clip_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, select_nodes_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=None, global_step=global_step)

    def forward(self, ob, select_nodes_ls, neighbor_map, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, policy, done, hidden):
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, policy, done)


class IA2C(A2C):
    def __init__(self, n_s, n_a, n_w, n_n, n_agent, total_step,
                 model_config, seed=0, checkpoint_dir= ""):
        # super(IA2C, self).__init__()
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = n_agent
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.value_coef = model_config.getfloat('value_coef')
        self.entropy_coef = model_config.getfloat('entropy_coef_init')
        self.opt_epochs = model_config.getint('opt_epochs')
        self.n_s = n_s
        self.n_a = n_a
        self.n_w = n_w
        self.n_n = n_n
        self.n_step = model_config.getint('batch_size')
        self.minibatch_steps = self.n_step//2
        self.policy = self._init_policy(n_s, n_a, n_w, n_n, model_config, agent_name='ppo')
        self.policy_old = copy.deepcopy(self.policy)
        self.agent_ls = []
        for i in range(n_agent):
            self.agent_ls.append(Agent(i))
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        # self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        lambd = model_config.getfloat('lambd')
        sample_size = model_config.getfloat('batch_size')
        self.trans_buffer_ls = []
        self.policy.prepare_loss()
        for i in range(self.n_agent):
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma, lambd, sample_size))

    def backward(self, progress, R_ls, select_nodes_ls, summary_writer=None, global_step=None):
        cur_lr_func = lambda a: 2.5e-4 * (1. - a)
        cur_beta_func = self.beta_scheduler.get()
        cur_clip_func = lambda a: .1 * (1. - a)
        E = self.opt_epochs
        obs_all = []
        acts_all = []
        masks_all = []
        Rs_all = []
        Advs_all = []
        hiddens_all = []
        messages_all = []
        for idx, node_idx in enumerate(select_nodes_ls):
            obs, acts, masks, Rs, Advs, hiddens, messages = self.trans_buffer_ls[node_idx].sample_transition(R_ls[idx])
            obs_all.append(obs)
            acts_all.append(acts)
            masks_all.append(masks)
            Rs_all.append(Rs)
            Advs_all.append(Advs)
            hiddens_all.append(hiddens)
            messages_all.append(messages)
        obs = np.concatenate(obs_all)
        acts = torch.concat(acts_all)
        masks = torch.concat(masks_all)
        Rs = torch.concat(Rs_all)
        Advs = torch.concat(Advs_all)
        hiddens = torch.concat(hiddens_all)
        messages = torch.concat(messages_all)

        self.policy_old.load_state_dict(self.policy.state_dict())
        for e in range(E):
            self.policy.zero_grad()
            b_inds = np.arange(self.n_step * self.n_agent)
            np.random.shuffle(b_inds)
            for start in range(0, self.n_step, self.minibatch_steps * self.n_agent):
                mb_inds = b_inds[start:start + (self.minibatch_steps * self.n_agent)]
                mb_obs, mb_masks, mb_actions, mb_advantages, mb_returns, mb_hiddens, mb_messages = \
                    [arr[mb_inds] for arr in [obs, masks, acts, Advs, Rs, hiddens, messages]]
                mb_pis, mb_vs = self.policy(mb_obs, mb_hiddens, mb_messages, 'pv')
                mb_pi_olds, mb_v_olds = self.policy_old(mb_obs, mb_hiddens, mb_messages,'pv')
                mb_pi_olds, mb_v_olds = mb_pi_olds.detach(), mb_v_olds.detach()

                self.policy.backward(cur_clip_func(progress),
                                                  mb_pis, mb_vs, mb_pi_olds, mb_v_olds,
                                                  mb_actions, mb_advantages, mb_returns,
                                                  cur_lr_func(progress), self.value_coef, self.entropy_coef)

    def forward(self, obs, select_nodes_ls, neighbor_map, done, message_from_who = None,out_type='pv'):
        if out_type == 'v':
            out = []
        elif out_type == 'p':
            out1, out2, out3 = [],[],[]
        else:
            out1, out2, out3, out4 = [], [], [], []
        pre_messages = [copy.copy(agent.message) for agent in self.agent_ls]
        for select_node in select_nodes_ls:
            hidden = self.agent_ls[select_node].hidden
            # hidden = hidden/self.n_agent
            if out_type == 'pv' or out_type == 'p':
                message = pre_messages[message_from_who[select_node]]
            else:
                message = torch.zeros((1,1))

            cur_out = self.policy(obs[select_node].reshape(1,-1), hidden.unsqueeze(0), message, out_type)
            if len(out_type) == 1:
                if out_type is 'v':
                    out.append(cur_out[0])
                else:
                    out1.append(cur_out[0][0])
                    out2.append(hidden)
                    out3.append(message)
                    self.agent_ls[select_node].update(cur_out[1], cur_out[2])
            else:
                out1.append(cur_out[0][0])
                out2.append(cur_out[1][0])
                out3.append(hidden)
                out4.append(message)
                self.agent_ls[select_node].update(cur_out[2], cur_out[3])
        if out_type == 'v':
            return out
        elif out_type == 'p':
            return out1, out2, out3
        else:
            return out1, out2, out3, out4

    def appendBuffer(self):
        for i in range(self.n_agent):
            if len(self.trans_buffer_ls[i].buffers) == 0:
                self.trans_buffer_ls[i].reset()
            else:
                self.trans_buffer_ls[i].reset(self.trans_buffer_ls[i].buffers[-1]['dones'][-1])

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for agent in self.agent_ls:
            agent._reset()

    def add_transition(self, obs, actions, rewards, values, done, hiddens, messages, select_nodes_ls):
        masks = torch.from_numpy((1. - np.array([done])))
        for idx, node in enumerate(select_nodes_ls):
            self.trans_buffer_ls[node].add_transition(obs[node], np.array([actions[idx]]),
                                                   np.array([rewards[node]]), values[idx],masks, hiddens[idx], messages[idx])





