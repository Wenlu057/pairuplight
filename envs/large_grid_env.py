

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from large_grid.data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave', 'wait']
PHASE_NUM = 4



class LargeGridPhase(PhaseMap):
    def __init__(self):

        phases = ['rrgrrrrrgrrr','ggrrrrggrrrr',
                  'rrrrrgrrrrrg','rrrggrrrrggr']
        self.phases = {PHASE_NUM: PhaseSet(phases)}


class LargeGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))


class LargeGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return PHASE_NUM

    def _init_large_neighbor_map(self):
        neighbor_map = {}
        # corner nodes
        neighbor_map['nt1'] = ['none', 'nt2', 'none', 'nt7']
        neighbor_map['nt6'] = ['nt5', 'none', 'none', 'nt12']
        neighbor_map['nt31'] = ['none', 'nt32', 'nt25', 'none']
        neighbor_map['nt36'] = ['nt35', 'none', 'nt30', 'none']
        # edge nodes
        neighbor_map['nt2'] = ['nt1','nt3', 'none','nt8']
        neighbor_map['nt3'] = ['nt2', 'nt4', 'none', 'nt9']
        neighbor_map['nt4'] = ['nt3', 'nt5', 'none', 'nt10']
        neighbor_map['nt5'] = ['nt4', 'nt6','none', 'nt11']
        neighbor_map['nt32'] = ['nt31', 'nt33', 'nt26', 'none']
        neighbor_map['nt33'] = ['nt32', 'nt34', 'nt27', 'none']
        neighbor_map['nt34'] = ['nt33', 'nt35', 'nt28', 'none']
        neighbor_map['nt35'] = ['nt34', 'nt36', 'nt29', 'none']
        neighbor_map['nt12'] = ['nt11', 'nt18', 'nt6', 'none']
        neighbor_map['nt18'] = ['nt17', 'none', 'nt12', 'nt24']
        neighbor_map['nt24'] = [ 'nt23', 'none', 'nt18','nt30']
        neighbor_map['nt30'] = [ 'nt29', 'none', 'nt24','nt36']
        neighbor_map['nt7'] = ['none', 'nt8', 'nt1', 'nt13']
        neighbor_map['nt13'] = ['none','nt14', 'nt7', 'nt19']
        neighbor_map['nt19'] = ['none', 'nt20', 'nt13', 'nt25']
        neighbor_map['nt25'] = ['none', 'nt26','nt19', 'nt31']
        # internal nodes
        for i in [8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29]:
            n_node = 'nt' + str(i + 6)
            s_node = 'nt' + str(i - 6)
            w_node = 'nt' + str(i - 1)
            e_node = 'nt' + str(i + 1)
            cur_node = 'nt' + str(i)
            neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
        return neighbor_map

    def _init_large_distance_map(self):
        distance_map = {}
        # corner nodes
        distance_map['nt1'] = ['none', 'none', 'none', 'none', 'nt3', 'none', 'nt8', 'nt13']
        distance_map['nt6'] = ['none', 'none', 'none', 'nt4', 'none', 'nt11', 'none', 'nt18']
        distance_map['nt31'] = ['nt19', 'none', 'nt26', 'none', 'nt33', 'none', 'none', 'none']
        distance_map['nt36'] = ['nt24', 'nt29', 'none', 'nt34', 'none', 'none', 'none', 'none']
        # edge nodes
        distance_map['nt2'] = ['none', 'none', 'none', 'none', 'nt4', 'nt7', 'nt9', 'nt14']
        distance_map['nt3'] = ['none', 'none', 'none', 'nt1', 'nt5', 'nt8', 'nt10', 'nt15']
        distance_map['nt4'] = ['none', 'none', 'none', 'nt2', 'nt6', 'nt9', 'nt11', 'nt16']
        distance_map['nt5'] = ['none', 'none', 'none', 'nt3', 'none', 'nt10', 'nt12', 'nt17']
        distance_map['nt32'] = ['nt20', 'nt25', 'nt27', 'none', 'nt34', 'none', 'none', 'none']
        distance_map['nt33'] = ['nt21', 'nt26', 'nt28', 'nt31', 'nt35', 'none', 'none', 'none']
        distance_map['nt34'] = ['nt22', 'nt27', 'nt29', 'nt32', 'nt36', 'none', 'none', 'none']
        distance_map['nt35'] = ['nt23', 'nt28', 'nt30', 'nt33', 'none', 'none', 'none', 'none']
        distance_map['nt12'] = ['none', 'nt5', 'none', 'nt10', 'none', 'nt17', 'none', 'nt24']
        distance_map['nt18'] = ['nt6', 'nt11', 'none', 'nt16', 'none', 'nt23', 'none', 'nt30']
        distance_map['nt24'] = ['nt12', 'nt17', 'none', 'nt22', 'none', 'nt29', 'none', 'nt36']
        distance_map['nt30'] = ['nt18', 'nt23', 'none', 'nt28', 'none', 'nt35', 'none', 'none']
        distance_map['nt7'] = ['none', 'none', 'none', 'nt5', 'nt9', 'none', 'nt14', 'nt19']
        distance_map['nt13'] = ['nt1', 'none', 'nt8', 'none', 'nt15', 'none', 'nt20', 'nt25']
        distance_map['nt19'] = ['nt7', 'none', 'nt14', 'none', 'nt21', 'none', 'nt26', 'nt31']
        distance_map['nt25'] = ['nt13', 'none', 'nt20', 'none', 'nt27', 'none', 'nt32', 'none']
        # internal nodes
        distance_map['nt8'] = ['none', 'nt1', 'nt3', 'none', 'nt10', 'nt13', 'nt15', 'nt20']
        distance_map['nt9'] = ['none', 'nt2', 'nt4', 'nt7', 'nt11', 'nt14', 'nt16', 'nt21']
        distance_map['nt10'] = ['none', 'nt3', 'nt5', 'nt8', 'nt12', 'nt15', 'nt17', 'nt22']
        distance_map['nt11'] = ['none', 'nt4', 'nt6', 'nt9', 'none', 'nt16', 'nt18', 'nt23']

        distance_map['nt14'] = ['nt2', 'nt7', 'nt9', 'none', 'nt16', 'nt19', 'nt21', 'nt26']
        distance_map['nt15'] = ['nt3', 'nt8', 'nt10', 'nt13', 'nt17', 'nt20', 'nt22', 'nt27']
        distance_map['nt16'] = ['nt4', 'nt9', 'nt11', 'nt14', 'nt18', 'nt21', 'nt23', 'nt28']
        distance_map['nt17'] = ['nt5', 'nt10', 'nt12', 'nt15', 'none', 'nt22', 'nt24', 'nt29']

        distance_map['nt20'] = ['nt8', 'nt13', 'nt15', 'none', 'nt22', 'nt25', 'nt27', 'nt32']
        distance_map['nt21'] = ['nt9', 'nt14', 'nt16', 'nt19', 'nt23', 'nt26', 'nt28', 'nt33']
        distance_map['nt22'] = ['nt10', 'nt15', 'nt17', 'nt20', 'nt24', 'nt27', 'nt29', 'nt34']
        distance_map['nt23'] = ['nt11', 'nt16', 'nt18', 'nt21', 'none', 'nt28', 'nt30', 'nt35']

        distance_map['nt26'] = ['nt14', 'nt19', 'nt21', 'none', 'nt28', 'nt31', 'nt33', 'none']
        distance_map['nt27'] = ['nt15', 'nt20', 'nt22', 'nt25', 'nt29', 'nt32', 'nt34', 'none']
        distance_map['nt28'] = ['nt16', 'nt21', 'nt23', 'nt26', 'nt30', 'nt33', 'nt35', 'none']
        distance_map['nt29'] = ['nt17', 'nt22', 'nt24', 'nt27', 'none', 'nt34', 'nt36', 'none']

        return distance_map

    def _init_map(self):
        self.neighbor_map = self._init_large_neighbor_map()
        # for spatial discount
        self.distance_map = self._init_large_distance_map()
        self.max_distance = 6
        self.phase_map = LargeGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
                            self.peak_flow2,
                            self.init_density,
                            seed=seed,
                            thread=self.sim_thread)


    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals, color=c, label=label)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_large.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = LargeGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(2)
    ob = env.reset()
    controller = LargeGridController(env.node_names)
    rewards = []
    while True:
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
