"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET
import random

DEFAULT_PORT = 8813
SEC_IN_MS = 1000

# hard code real-net reward norm
REALNET_REWARD_NORM = 20

class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]




class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        # self.edges_in = []  # for reward
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.last_reward = []
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()
        self.terminate()
        self.total_arrived=0
        self.total_loaded = 0

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(ild)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(ild)
                # cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        if phase_type == 'green':
            cur_phase = self.phase_map.get_phase(node.phase_id, action)
            node.prev_action = action
            return cur_phase, action


    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        # wait / wave states for each lane
        return len(node.lanes_in)


    def _get_state(self):
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            cur_state = [node.wave_state]


            # include wait state
            if 'wait' in self.state_names:
                cur_state.append(node.wait_state)

            # add 2 hop neighbor after the wait
            visited_2hop = []
            for nnode_name in node.neighbor:
                neighbor_wave_state = []
                if nnode_name != 'none':
                    neighbor_node = self.nodes[nnode_name]
                    cur_state.append(neighbor_node.wave_state * self.coop_gamma)
                else:
                    cur_state.append(np.zeros(node.num_state))

            for hop2nnode_name in self.distance_map[node_name]:
                if hop2nnode_name != 'none':
                    hop2_node = self.nodes[hop2nnode_name]
                    cur_state.append(hop2_node.wave_state)
                else:
                    cur_state.append(np.zeros(node.num_state))

            state.append(np.concatenate(cur_state))
        return state

    def _init_nodes(self):
        nodes = {}
        for idx in range(self.sim.trafficlight.getIDCount()):
            node_name = 'nt' + str(idx + 1)
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                neighbor = []
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in
            nodes[node_name].links_in = self.sim.trafficlight.getControlledLinks(node_name)
            ilds_in = []
            for lane_name in lanes_in:
                ild_name = lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            # nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = list(nodes.keys())
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            s += '\tilds_in: %r\n' % node.ilds_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a= 0
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            self.n_a = node.n_a


    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num))
        return policy

    def _init_sim(self, seed, gui=False):
        sumocfg_file = "large_grid/data/exp_6by6.sumocfg"
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]

        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport', '600'] # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # command += ['--waiting-time-memory', '3600']
        # collect trip info if necessary
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()

        self.n_s = self.nodes[self.node_names[0]].num_state
        self.n_n = self.nodes[self.node_names[0]].num_state * 12
        self.n_w = self.nodes[self.node_names[0]].num_state


    def get_most_congested(self, threshold=1.8):
        car_wait_queues = []
        for i, node_name in enumerate(self.node_names):
            max_car_wait = 0
            for ild in self.nodes[node_name].ilds_in:
                cur_cars = self.sim.lanearea.getLastStepVehicleIDs('oppo_' +str(ild))
                max_pos_vid = ''
                max_pos = 0
                for vid in cur_cars:
                    car_pos = self.sim.vehicle.getLanePosition(vid)
                    if car_pos > max_pos:
                        max_pos = car_pos
                        max_pos_vid = vid
                car_wait = self.sim.vehicle.getWaitingTime(max_pos_vid) if max_pos_vid != '' else 0
                max_car_wait = max(max_car_wait, car_wait)

                if max_car_wait >= 30:
                    car_wait_queues.append(i)
                    break

        return car_wait_queues

    def _measure_reward_step(self):
        rewards = []
        waiting = []


        for i, node_name in enumerate(self.node_names):
            queues = []
            waits = []
            totoal_waits = 0
            total_left_cars = 0
            for ild in self.nodes[node_name].ilds_in:
                if self.obj in ['queue', 'hybrid']:
                    if self.name == 'real_net':
                        cur_queue = min(10, self.sim.lane.getLastStepVehicleNumber(ild))
                    else:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber("oppo_"+ str(ild))
                        queues.append(cur_queue)

                if self.obj in ['wait', 'hybrid']:
                    max_pos = 0
                    car_wait = 0
                    if self.name == 'real_net':
                        cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)
                    else:
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs("oppo_"+ str(ild))
                        # totoal_cars += len(cur_cars)
                    for vid in cur_cars:
                        # car_wait += self.sim.vehicle.getWaitingTime(vid)
                        car_pos = self.sim.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.sim.vehicle.getWaitingTime(vid)
                    waits.append(car_wait)


            queue = np.sum(np.array(queues)) if len(queues) else 0
            wait = np.max(np.array(waits)) if len(waits) else 0

            if self.obj == 'queue':
                # reward = - queue
                reward = 0
            elif self.obj == 'wait':
                clip_wait = np.clip(wait, 0, 500)
                reward = - clip_wait

            else:
                reward = - queue - self.coef_wait * wait
                clip_wait = np.clip(wait, 0, 500)
                reward = -queue - clip_wait * 0.1
                # reward = (totoal_cars - total_wait_cars)/totoal_cars if totoal_cars != 0 else 1

            rewards.append(reward)
            waiting.append(-wait)
        return np.array(rewards), np.array(waiting)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for link in node.links_in:
                        arrive_vehs = self.sim.lanearea.getLastStepVehicleNumber("oppo_" + str(link[0][0]))
                        departure_vehs = self.sim.lanearea.getLastStepVehicleNumber(link[0][1])
                        cur_state.append(arrive_vehs - departure_vehs)
                    cur_state = np.array(cur_state)
                else:
                    cur_state = []
                    head_car_id = []
                    for idx, ild in enumerate(node.ilds_in):
                        max_pos = 0
                        car_wait = 0
                        car_signal = -1
                        car_id = ''
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs("oppo_"+str(ild))
                        for vid in cur_cars:
                            car_pos = self.sim.vehicle.getLanePosition(vid)
                            if car_pos > max_pos:
                                max_pos = car_pos
                                car_id = vid
                        if car_id != '':
                            car_wait = self.sim.vehicle.getWaitingTime(car_id)
                            car_signal = self.sim.vehicle.getSignals(car_id)
                        if idx == 0 or idx == 3:
                            if car_signal == 10:
                                cur_state.append(0)
                                cur_state.append(0)
                                cur_state.append(car_wait)
                            else:
                                cur_state.append(car_wait)
                                cur_state.append(car_wait)
                                cur_state.append(0)
                        elif idx == 1 or idx == 4:
                            cur_state.append(car_wait)
                            cur_state.append(car_wait)
                        else:
                            cur_state.append(car_wait)

                    cur_state = np.array(cur_state)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                if state_name == 'wave':
                    node.wave_state = norm_cur_state
                else:
                    node.wait_state = norm_cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                queues.append(self.sim.lane.getLastStepHaltingNumber(ild))
        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = np.round(x / norm, decimals=3)
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            node.num_fingerprint = node.n_a - 1
            node.num_state = self._get_node_state_num(node)


    def _set_phase(self, select_action_ls, select_nodes_ls, phase_type, phase_duration):
        action_node_pair_ls = {}
        new_action = []
        if len(select_nodes_ls) != 0:
            for a, node_idx in zip(select_action_ls, select_nodes_ls):
                action_node_pair_ls[self.node_names[node_idx]] = a
        for node_name in self.node_names:
            if node_name in action_node_pair_ls.keys():
                phase, action = self._get_node_phase(action_node_pair_ls[node_name], node_name, phase_type)
                new_action.append(action)
            else:
                phase, _ = self._get_fixed_time_node_phase(node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)
        return new_action

    def _get_fixed_time_node_phase(self, node_name, phase_type):
        node = self.nodes[node_name]
        action = (node.prev_action + 1) % node.phase_id
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        node.prev_action = action
        return cur_phase, action

    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            self.total_arrived += self.sim.simulation.getArrivedNumber()
            self.total_loaded += self.sim.simulation.getDepartedNumber()
            self.cur_sec += 1
            if self.total_loaded == self.total_arrived:
                return True
            if self.is_record:
                self._measure_traffic_step()
        # return reward
        return False

    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                lane = node.ilds_in[i]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        self.total_arrived = 0
        self.total_loaded = 0
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        # initialize fingerprint
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        # next environment random condition should be different
        self.seed += 1
        return self._get_state()

    def terminate(self):
        self.sim.close()

    def step(self, action, select_node_idx):
        if self.agent == 'a2c':
            action = self._transfer_action(action)

        action = self._set_phase(action, select_node_idx, 'green', self.control_interval_sec)
        done = self._simulate(self.control_interval_sec)

        state = self._get_state()
        reward, performance = self._measure_reward_step()

        global_reward = np.average(performance) # for fair comparison

        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_r,
                           'reward': global_reward}
            self.control_data.append(cur_control)

        # use local rewards in test
        if not self.train_mode:
            return state, performance, done, global_reward

        return state, reward, done, global_reward

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]

    def whose_message(self):
        message_from = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            neighbor_wait_state = {}
            node_waiting = 0
            neighbor_wait_state[node_name] = max(node.wait_state)

            for i, nnode_name in enumerate(node.neighbor):
                if nnode_name != 'none':
                    neighbor_node = self.nodes[nnode_name]
                    neighbor_waiting = 0
                    for link in neighbor_node.links_in:
                        if link[0][1] in node.ilds_in:

                            neighbor_waiting = max(neighbor_node.wait_state[neighbor_node.links_in.index(link)], neighbor_waiting)

                    neighbor_wait_state[nnode_name] = neighbor_waiting
            node_idx = self.node_names.index(max(neighbor_wait_state, key=neighbor_wait_state.get))
            message_from.append(node_idx)
        return message_from
