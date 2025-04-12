import itertools
import logging
import numpy as np
import tensorflow as tf
import torch
import time
import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])

def mean_std_groups(x, y, group_size):
    num_groups = int(len(x) / group_size)

    x, x_tail = x[:group_size * num_groups], x[group_size * num_groups:]
    x = x.reshape((num_groups, group_size))

    y, y_tail = y[:group_size * num_groups], y[group_size * num_groups:]
    y = y.reshape((num_groups, group_size))

    x_means = x.mean(axis=1)
    x_stds = x.std(axis=1)

    if len(x_tail) > 0:
        x_means = np.concatenate([x_means, x_tail.mean(axis=0, keepdims=True)])
        x_stds = np.concatenate([x_stds, x_tail.std(axis=0, keepdims=True)])

    y_means = y.mean(axis=1)
    y_stds = y.std(axis=1)

    if len(y_tail) > 0:
        y_means = np.concatenate([y_means, y_tail.mean(axis=0, keepdims=True)])
        y_stds = np.concatenate([y_stds, y_tail.std(axis=0, keepdims=True)])

    return x_means, x_stds, y_means, y_stds


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        # if self.init_test:
        #     self.init_test = False
        #     return True
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    # def update_test(self, reward):
    #     if self.prev_reward is not None:
    #         if abs(self.prev_reward - reward) <= self.delta_reward:
    #             self.stop = True
    #     self.prev_reward = reward

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        # self.sess = self.model.sess
        self.n_step = self.model.n_step
        # self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        # self._init_summary()
        self.plot_points = 10
        self.steps_histr = []
        self.reward_histr = []
        self.epochs = 0

    def _init_summary(self):
        self.train_reward = tf.compat.v1.placeholder(tf.float32, [])
        self.train_summary = tf.compat.v1.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.compat.v1.placeholder(tf.float32, [])
        self.test_summary = tf.compat.v1.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done, select_nodes_ls, waiting_threshold=30):
        ob = prev_ob
        done = prev_done
        rewards = []
        agent_level_reward = []
        for i in range(self.n_step):  # n_step is the T(worker steps)
            # self.env.total_arrived = 0
            message_from_who = self.env.whose_message()
            policy, value, hidden, message = self.model.forward(ob, select_nodes_ls, self.env.neighbor_map, done, message_from_who)
            action = []
            for pi in policy:
                # action.append(np.random.choice(np.arange(len(pi)), p=pi))
                epsilon = 0.15
                if self.global_counter.cur_step > 10000:
                    epsilon = 0.1
                if np.random.rand() > epsilon:
                    action.append(np.argmax(pi.detach().numpy()))
                else:
                    action.append(np.random.choice(4))

            next_ob, reward, done, global_reward = self.env.step(action, select_nodes_ls)
            rewards.append(global_reward)
            agent_level_reward.append(reward)
            self.global_counter.next()
            self.cur_step += 1
            # if self.global_counter.cur_step < self.global_counter.total_step // 2:
            #     policy = [p.detach() for p in policy]
            self.model.add_transition(ob, action, reward, value, done, hidden, message, select_nodes_ls)

            # logging
            if self.cur_step % 10 == 0:
                logging.info('''Training: episode step %d, a: %s, r: %.2f, done: %r''' %
                             (self.cur_step, str(action), global_reward, done))
            ob = next_ob
            if done:
                break
            ob = next_ob
        if self.agent.endswith('a2c'):
            if done:
                R = []
                end_v = np.average(np.array(self.rewards))
                for i in select_nodes_ls:
                    # R.append(torch.tensor(np.clip(end_v + 100, -20, 20)).unsqueeze(0))
                    R.append(torch.tensor(0).unsqueeze(0))
                # R = 0 if self.agent == 'a2c' else [(np.average(np.array(rewards)) - fixed_time_control_performance)/350] * len(select_nodes_ls)
            else:
                R = self.model.forward(ob, select_nodes_ls, self.env.neighbor_map, False, out_type='v')
                # R = []
                # for i in select_nodes_ls:
                #     agent_reward = []
                #     for idx in range(len(agent_level_reward)):
                #         if idx != 0:
                #             agent_reward.append(agent_level_reward[idx][i] - agent_level_reward[idx - 1][i])
                #     if (np.std([agent_level_reward[idx][i],agent_level_reward[idx - 1][i]])) < 5:
                #     # if np.std(agent_reward) < 100:
                #         if np.sum(np.array(agent_reward)) > 0:
                #             R.append(torch.tensor([10]))
                #         else:
                #             R.append(torch.tensor([-10]))
                #     else:
                #         end_v = (np.sum(np.array(agent_reward))/(self.n_step - 1))
                #         R.append(torch.tensor(np.clip(end_v, -10,10)).unsqueeze(0))

        else:
            R = 0
        return ob, done, R, rewards



    def perform(self, policy_type='deterministic'):
        ob = self.env.reset(gui=False)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c'):
                # policy-based on-poicy learning
                # select_nodes_ls = self.env.get_most_congested()
                select_nodes_ls = np.arange(36)
                message_from_who = self.env.whose_message()
                policy, hidden, message = self.model.forward(ob, select_nodes_ls, self.env.neighbor_map, done, message_from_who, 'p')
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    if policy_type != 'deterministic':
                        action = np.random.choice(np.arange(len(policy)), p=policy)
                    else:
                        action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        if policy_type != 'deterministic':
                            action.append(np.random.choice(np.arange(len(pi)), p=pi))
                        else:
                            action.append(np.argmax(pi.detach().numpy()))
            else:
                # value-based off-policy learning
                if policy_type != 'stochastic':
                    action, _ = self.model.forward(ob)
                else:
                    action, _ = self.model.forward(ob, stochastic=True)
            next_ob, reward, done, global_reward = self.env.step(action, select_nodes_ls)
            rewards.append(global_reward)
            if done or len(rewards) > 720:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        sum_reward = np.sum(np.array(rewards))
        return mean_reward, std_reward, sum_reward

    def run(self):
        losses = []
        test_reward = []
        self.model.reset()
        while not self.global_counter.should_stop():
            global_step = self.global_counter.cur_step
            # if global_step >= self.global_counter.total_step//2:
            # if True:
            #     self.env.train_mode = False
            #     mean_reward, std_reward, sum_reward = self.perform()
            #     self.env.terminate()
            #     log = {'agent': self.agent,
            #            'step': global_step,
            #            'avg_reward': mean_reward,
            #            'std_reward': std_reward,
            #            'sum_reward': sum_reward}
            #     test_reward.append(log)
            #     df = pd.DataFrame(test_reward)
            #     df.to_csv(self.output_path + 'test_reward.csv')
            #     self.steps_histr.append(global_step)
            #     self.reward_histr.append(sum_reward)
            #     # self._add_summary(mean_reward, global_step, is_train=False)
            #     logging.info('Testing: global step %d, avg R: %.2f' %
            #                  (global_step, sum_reward))
            #     # statistic logic
            #     group_size = len(self.steps_histr) // self.plot_points
            #     if len(self.steps_histr) % self.plot_points == 0 and group_size >= 4:
            #         x_means, _, y_means, y_stds = \
            #             mean_std_groups(np.array(self.steps_histr), np.array(self.reward_histr), group_size)
            #         fig = plt.figure()
            #         plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 6))
            #         plt.errorbar(x_means, y_means, yerr=y_stds, ecolor='xkcd:blue', fmt='xkcd:black', capsize=5,
            #                      elinewidth=1.5,
            #                      mew=1.5, linewidth=1.5)
            #         plt.title('Training progress')
            #         plt.xlabel('Total steps')
            #         plt.ylabel('Episode reward')
            #         plt.savefig(self.output_path + '/episode_reward.png', dpi=200)
            #         plt.clf()
            #         plt.close()
            #         plot_timer = 0
            #         fig.set_size_inches(8, 6)

            # train
            self.env.train_mode = True
            ob = self.env.reset(gui=False)

            # ob = torch.tensor(self.env.reset())
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.rewards = []
            self.cur_step = 0

            while True:
                progress = self.global_counter.cur_step / self.global_counter.total_step
                # select_nodes_ls = self.env.get_most_congested()
                select_nodes_ls = np.arange(36)
                # for node_idx in select_nodes_ls:
                #     if node_idx not in self.model.trained_policy_ls:
                #         self.model.trained_policy_ls.append(node_idx)
                if len(select_nodes_ls) == 0:
                    next_ob, reward, done, global_reward = self.env.step(0, select_nodes_ls)
                    # self.model.last_reward_ls = reward
                    ob = next_ob
                    self.cur_step += 1
                    self.rewards.append(global_reward)
                    # global_step = self.global_counter.next()
                    if done:
                        self.env.terminate()
                        break
                    continue
                ob, done, R, cur_rewards = self.explore(ob, done, select_nodes_ls)
                self.rewards += cur_rewards
                # global_step = self.global_counter.cur_step
                if self.agent.endswith('a2c'):
                    if not done:
                        self.model.backward(progress, R, select_nodes_ls, None, None)
                else:
                    self.model.backward(None, None)
                # termination
                if done:
                    self.env.terminate()
                    break
            self.epochs += 1
            rewards = np.array(self.rewards)
            sum_reward = np.sum(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': self.epochs,
                   'throughput': self.cur_step,
                   'test_id': -1,
                   'sum_reward': sum_reward,
                   'avg_reward': np.average(rewards),
                   'std_reward': std_reward}
            self.data.append(log)
            # self._add_summary(mean_reward, global_step)
            # self.summary_writer.flush()
            logging.info('''Training: global step %d, total_timestep: %d, sum_reward: %.2f''' %
                         (self.epochs, self.cur_step, sum_reward))
            df = pd.DataFrame(self.data)
            df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, demo=self.demo, policy_type=self.policy_type)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
