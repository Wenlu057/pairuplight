# -*- coding: utf-8 -*-
"""
build *.xml files for a large 5 x 5 network
w/ the traffic dynamics modified from the following paper:

Chu, Tianshu, Shuhui Qu, and Jie Wang. "Large-scale traffic grid signal control with
regional reinforcement learning." American Control Conference (ACC), 2016. IEEE, 2016.

@author: Tianshu Chu
"""
import numpy as np
import os

MAX_CAR_NUM = 30
SPEED_LIMIT_ST = 20
SPEED_LIMIT_AV = 11
L0 = 200
L0_end = 75
N = 5


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def output_nodes(node):
    str_nodes = '<nodes>\n'
    # traffic light nodes
    ind = 1
    for dy in np.arange(0, L0 * 6, L0):
        for dx in np.arange(0, L0 * 6, L0):
            str_nodes += node % ('nt' + str(ind), dx, dy, 'traffic_light')
            ind += 1
    # other nodes
    ind = 1
    for dx in np.arange(0, L0 * 6, L0):
        str_nodes += node % ('np' + str(ind), dx, -L0_end, 'priority')
        ind += 1
    for dy in np.arange(0, L0 * 6, L0):
        str_nodes += node % ('np' + str(ind), L0 * 5 + L0_end, dy, 'priority')
        ind += 1
    for dx in np.arange(L0 * 5, -1, -L0):
        str_nodes += node % ('np' + str(ind), dx, L0 * 5 + L0_end, 'priority')
        ind += 1
    for dy in np.arange(L0 * 5, -1, -L0):
        str_nodes += node % ('np' + str(ind), -L0_end, dy, 'priority')
        ind += 1
    str_nodes += '</nodes>\n'
    return str_nodes


def output_road_types():
    str_types = '<types>\n'
    str_types += '  <type id="a" priority="2" numLanes="2" speed="%.2f"/>\n' % SPEED_LIMIT_ST
    str_types += '  <type id="b" priority="1" numLanes="1" speed="%.2f"/>\n' % SPEED_LIMIT_AV
    str_types += '</types>\n'
    return str_types


def get_edge_str(edge, from_node, to_node, edge_type):
    edge_id = '%s_%s' % (from_node, to_node)
    return edge % (edge_id, from_node, to_node, edge_type)


def output_edges(edge):
    str_edges = '<edges>\n'
    # external roads
    in_edges = [6, 12, 18, 24, 30, 36, 31, 25, 19, 13, 7, 1]
    out_edges = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'a')
        str_edges += get_edge_str(edge, out_node, in_node, 'a')

    in_edges = [1, 2, 3, 4, 5, 6, 36, 35, 34, 33, 32, 31]
    out_edges = [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'b')
        str_edges += get_edge_str(edge, out_node, in_node, 'b')
    # internal roads
    for i in range(1, 36, 6):
        for j in range(5):
            from_node = 'nt' + str(i + j)
            to_node = 'nt' + str(i + j + 1)
            str_edges += get_edge_str(edge, from_node, to_node, 'a')
            str_edges += get_edge_str(edge, to_node, from_node, 'a')
    for i in range(1, 7):
        for j in range(0, 30, 6):
            from_node = 'nt' + str(i + j)
            to_node = 'nt' + str(i + j + 6)
            str_edges += get_edge_str(edge, from_node, to_node, 'b')
            str_edges += get_edge_str(edge, to_node, from_node, 'b')
    str_edges += '</edges>\n'
    return str_edges


def get_con_str(con, from_node, cur_node, to_node, from_lane, to_lane):
    from_edge = '%s_%s' % (from_node, cur_node)
    to_edge = '%s_%s' % (cur_node, to_node)
    return con % (from_edge, to_edge, from_lane, to_lane)


def get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node):
    str_cons = ''
    # go-through
    str_cons += get_con_str(con, s_node, cur_node, n_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, w_node, 0, 0)
    # left-turn
    str_cons += get_con_str(con, s_node, cur_node, w_node, 0, 1)
    str_cons += get_con_str(con, n_node, cur_node, e_node, 0, 1)
    str_cons += get_con_str(con, w_node, cur_node, n_node, 1, 0)
    str_cons += get_con_str(con, e_node, cur_node, s_node, 1, 0)
    # right-turn
    str_cons += get_con_str(con, s_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, w_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, n_node, 0, 0)
    return str_cons


def output_connections(con):
    str_cons = '<connections>\n'
    # edge nodes
    in_edges = [6, 12, 18, 24, 30, 36, 31, 25, 19, 13, 7, 1]
    out_edges = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24]
    for i, j in zip(in_edges, out_edges):
        if i == 6:
            s_node = 'np6'
        elif i == 1:
            s_node = 'np1'
        else:
            s_node = 'nt' + str(i - 6)
        if i == 36:
            n_node = 'np13'
        elif i == 31:
            n_node = 'np18'
        else:
            n_node = 'nt' + str(i + 6)
        if i % 6 == 1:
            w_node = 'np' + str(j)
        else:
            w_node = 'nt' + str(i - 1)
        if i % 6 == 0:
            e_node = 'np' + str(j)
        else:
            e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    in_edges = [2, 3, 4, 5, 35, 34, 33, 32]
    out_edges = [2, 3, 4, 5, 14, 15, 16, 17]
    for i, j in zip(in_edges, out_edges):
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        if i <= 6:
            s_node = 'np' + str(j)
        else:
            s_node = 'nt' + str(i - 6)
        if i >= 30:
            n_node = 'np' + str(j)
        else:
            n_node = 'nt' + str(i + 6)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    # internal nodes
    for i in [8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29]:
        n_node = 'nt' + str(i + 6)
        s_node = 'nt' + str(i - 6)
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    str_cons += '</connections>\n'
    return str_cons


def output_netconfig():
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <edge-files value="exp_6by6.edg.xml"/>\n'
    str_config += '    <node-files value="exp_6by6.nod.xml"/>\n'
    str_config += '    <type-files value="exp_6by6.typ.xml"/>\n'
    str_config += '    <tllogic-files value="exp_6by6.tll.xml"/>\n'
    str_config += '    <connection-files value="exp_6by6.con.xml"/>\n'
    str_config += '  </input>\n  <output>\n'
    str_config += '    <output-file value="exp_6by6.net.xml"/>\n'
    str_config += '  </output>\n</configuration>\n'
    return str_config


def get_external_od(out_edges, dest=True):
    edge_maps = [0, 1, 2, 3, 4, 5, 6, 6, 12, 18, 24, 30, 36,
                 36, 35, 34, 33, 32, 31, 31, 25, 19, 13, 7, 1]
    cur_dest = []
    for out_edge in out_edges:
        in_edge = edge_maps[out_edge]
        in_node = 'nt' + str(in_edge)
        out_node = 'np' + str(out_edge)
        if dest:
            edge = '%s_%s' % (in_node, out_node)
        else:
            edge = '%s_%s' % (out_node, in_node)
        cur_dest.append(edge)
    return cur_dest


def sample_od_pair(orig_edges, dest_edges):
    from_edges = []
    to_edges = []
    for i in range(len(orig_edges)):
        from_edges.append(np.random.choice(orig_edges[i]))
        to_edges.append(np.random.choice(dest_edges))
    return from_edges, to_edges


def init_routes(density):
    init_flow = '  <flow id="i_%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="%d" departSpeed="0" number="%d" type="type1"/>\n'
    output = ''
    in_nodes = [6, 12, 18, 24, 30, 36, 31, 25, 19, 13, 7, 1,
                1, 2, 3, 4, 5, 6, 36, 35, 34, 33, 32, 31]
    out_nodes = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24,
                 1, 2, 3, 4, 5, 6,  13, 14, 15, 16, 17, 18]
    # external edges
    sink_edges = []
    for i, j in zip(in_nodes, out_nodes):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        sink_edges.append('%s_%s' % (node1, node2))

    def get_od(node1, node2, k, lane=0):
        source_edge = '%s_%s' % (node1, node2)
        sink_edge = np.random.choice(sink_edges)
        return init_flow % (str(k), source_edge, sink_edge, lane, car_num)

    # streets
    k = 1
    car_num = int(MAX_CAR_NUM * density)
    for i in range(1, 36, 6):
        for j in range(5):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 1)
            output += get_od(node1, node2, k)
            k += 1
            output += get_od(node2, node1, k)
            k += 1
            output += get_od(node1, node2, k, lane=1)
            k += 1
            output += get_od(node2, node1, k, lane=1)
            k += 1
    # avenues
    for i in range(1, 7):
        for j in range(0, 30, 6):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 6)
            output += get_od(node1, node2, k)
            k += 1
            output += get_od(node2, node1, k)
            k += 1
    return output

def output_flows(peak_flow1, peak_flow2, density, seed=None):
    '''
    flow1: x11, x12, x13, x14, x15 -> x1, x2, x3, x4, x5
    flow2: x16, x17, x18, x19, x20 -> x6, x7, x8, x9, x10
    flow3: x1, x2, x3, x4, x5 -> x15, x14, x13, x12, x11
    flow4: x6, x7, x8, x9, x10 -> x20, x19, x18, x17, x16
    '''
    if seed is not None:
        np.random.seed(seed)
    ext_flow = '  <flow id="f_%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="6" accel="6" decel="12"/>\n'
    # initial traffic dist
    if density > 0:
        str_flows += init_routes(density)

    # create external origins and destinations for flows
    srcs = []
    srcs.append(get_external_od([14, 15, 16, 17], dest=False))
    srcs.append(get_external_od([19, 21, 22, 24], dest=False))
    srcs.append(get_external_od([2, 3, 4, 5], dest=False))
    srcs.append(get_external_od([7, 9, 10, 12], dest=False))

    sinks = []
    sinks.append(get_external_od([5, 3, 4, 2]))
    sinks.append(get_external_od([7, 10, 9, 12]))
    sinks.append(get_external_od([17, 15, 16, 14]))
    sinks.append(get_external_od([19, 22, 21, 24]))

    # create volumes per 5 min for flows
    ratios1 = np.array([0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25]) # start from 0
    ratios2 = np.array([0.3, 0.8, 0.9, 1.0, 0.8, 0.6, 0.2])   # start from 15min
    flows1 = peak_flow1 * 0.6 * ratios1
    flows2 = peak_flow1 * ratios1
    flows3 = peak_flow2 * 0.6 * ratios2
    flows4 = peak_flow2 * ratios2
    flows = [flows1, flows2, flows3, flows4]
    times = np.arange(0, 3001, 300)
    id1 = len(flows1)
    id2 = len(times) - 1 - id1
    for i in range(len(times) - 1):
        name = str(i)
        t_begin, t_end = times[i], times[i + 1]
        # external flow
        k = 0
        if i < id1:
            for j in [0, 1]:
                for e1, e2 in zip(srcs[j], sinks[j]):
                    cur_name = name + '_' + str(k)
                    str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, flows[j][i])
                    k += 1
        if i >= id2:
            for j in [2, 3]:
                for e1, e2 in zip(srcs[j], sinks[j]):
                    cur_name = name + '_' + str(k)
                    str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, flows[j][i - id2])
                    k += 1
    str_flows += '</routes>\n'
    return str_flows


def gen_rou_file(path, peak_flow1, peak_flow2, density, seed=None, thread=None):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)
    write_file(path + flow_file, output_flows(peak_flow1, peak_flow2, density, seed=seed))
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_config(thread=None):
    # if thread is None:
    #     out_file = 'exp.rou.xml'
    # else:
    #     out_file = 'exp_%d.rou.xml' % int(thread)
    out_file = 'exp_6by6.rou.xml'
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="exp_6by6.net.xml"/>\n'
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="exp_6by6.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def get_ild_str(from_node, to_node, ild_str, lane_i=0):
    edge = '%s_%s' % (from_node, to_node)
    return ild_str % (edge, lane_i, edge, lane_i)


def output_ild(ild):
    str_adds = '<additional>\n'
    in_edges = [6, 12, 18, 24, 30, 36, 31, 25, 19, 13, 7, 1,
                1, 2, 3, 4, 5, 6, 36, 35, 34, 33, 32, 31]
    out_edges = [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24,
                 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18]
    # external edges
    for k, (i, j) in enumerate(zip(in_edges, out_edges)):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        str_adds += get_ild_str(node2, node1, ild)
        if k < 12:
            # streets
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
    # streets
    for i in range(1, 36, 6):
        for j in range(5):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 1)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
            str_adds += get_ild_str(node1, node2, ild, lane_i=1)
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
    # avenues
    for i in range(1, 7):
        for j in range(0, 30, 6):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 6)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
    str_adds += '</additional>\n'
    return str_adds


def output_tls(tls, phase):
    str_adds = '<additional>\n'
    # all crosses have 3 phases
    three_phases = ['rrgrrrrrgrrr', 'rryrrrrryrrr', 'rrrrrrrrrrrr',
                    'ggrrrrggrrrr', 'yyrrrryyrrrr', 'rrrrrrrrrrrr',
                    'rrrrrgrrrrrg', 'rrrrryrrrrry', 'rrrrrrrrrrrr',
                    'rrrggrrrrggr', 'rrryyrrrryyr', 'rrrrrrrrrrrr']
    phase_duration = [10, 4, 2,50,4,2]
    for i in range(1, 37):
        node = 'nt' + str(i)
        str_adds += tls % node
        for k, p in enumerate(three_phases):
            str_adds += phase % (phase_duration[k % 6], p)
        str_adds += '  </tlLogic>\n'
    str_adds += '</additional>\n'
    return str_adds


def main():
    # # nod.xml file
    # node = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
    # write_file('./exp_6by6.nod.xml', output_nodes(node))
    #
    # # typ.xml file
    # write_file('./exp_6by6.typ.xml', output_road_types())

    # # edg.xml file
    # edge = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
    # write_file('./exp_6by6.edg.xml', output_edges(edge))
    #
    # # con.xml file
    # con = '  <connection from="%s" to="%s" fromLane="%d" toLane="%d"/>\n'
    # write_file('./exp_6by6.con.xml', output_connections(con))

    # # tls.xml file
    # tls = '  <tlLogic id="%s" programID="0" offset="0" type="static">\n'
    # phase = '    <phase duration="%d" state="%s"/>\n'
    # write_file('./exp_6by6.tll.xml', output_tls(tls, phase))
    #
    # # net config file
    # write_file('./exp_6by6.netccfg', output_netconfig())

    # generate net.xml file
    # os.system('netconvert -c exp_6by6.netccfg')

    # raw.rou.xml file
    write_file('./exp_6by6.rou.xml', output_flows(800, 500, 0.1))

    # generate rou.xml file
    # os.system('jtrrouter -n exp.net.xml -r exp.raw.rou.xml -o exp.rou.xml')

    # add.xml file
    # ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s_%d" lane="%s_%d" pos="-50" endPos="-1"/>\n'
    # ild_in = '  <inductionLoop file="ild_out.out" freq="4" id="ild_in:%s_%d" lane="%s_%d" pos="-50"/>\n'
    # write_file('./exp_6by6_e1.add.xml', output_ild(ild_in))
    #
    # # config file
    # write_file('./exp_6by6.sumocfg', output_config())

if __name__ == '__main__':
    main()
