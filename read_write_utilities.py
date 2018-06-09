import pickle
import os
import numpy as np


#Reading labeled general graphs in the standard provided on https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
#We refer to the explanation https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets#file_format for details
def read_standard_graph(folderpath, filename):
    
    filename_edges = folderpath + '/' + filename + '/' + filename + '_A.txt'
    filename_nodes_to_graph = folderpath + '/' + filename + '/' + filename + '_graph_indicator.txt'
    filename_node_labels = folderpath + '/' + filename + '/' + filename + '_node_labels.txt'
    filename_classes = folderpath + '/' + filename + '/' + filename + '_graph_labels.txt'
    
    print(filename_edges)
    
    with open(filename_classes) as f_classes:
        classes = f_classes.read().splitlines()    
    nr_graphs = len(classes) + 1
    Vs, nodes_to_graph, node_labels, set_labels = read_node_labels(filename_nodes_to_graph, filename_node_labels, nr_graphs)
    Es = read_edges(filename_edges, Vs, nodes_to_graph, node_labels, nr_graphs)
    
    return Vs, Es, classes, set_labels

def read_node_labels(filename_nodes_to_graph, filename_node_labels, nr_graphs):
    with open(filename_nodes_to_graph) as f_nodes:
        nodes = f_nodes.read().splitlines()
        
    with open(filename_node_labels) as f_labels:
        labels = f_labels.read().splitlines()
    
    if (len(nodes) != len(labels)):
        raise ValueError('Node lists of different length')
        return -1
    
    Vs = [{} for _ in range(nr_graphs)]
    nodes_to_graph = {}
    node_labels = {}
    set_labels = set()
    for i in range(len(nodes)):
        node_id = i+1
        graph_id = int(nodes[i])
        nodes_to_graph[node_id] = graph_id
        label = labels[i]
        set_labels.add(label)
        node_labels[node_id] = label
        Vs[graph_id][node_id] = label
    return Vs, nodes_to_graph, node_labels, set_labels
        

def read_edges(filename_edges, Vs, nodes_to_graph, node_labels, nr_graphs, sep=','):
    Es = [{} for _ in range(nr_graphs)]
    f_edges = open(filename_edges, 'r')
    for line in f_edges: 
        line_split = str.split(line, sep)
        e1 = int(line_split[0].strip())
        e2 = int(line_split[1].strip())
        if (nodes_to_graph[e1] != nodes_to_graph[e2]):
            print('Vertices connected by and edge but belonging to different graphs')
            print('nodes',  e1, e2)
            print('graphs', nodes_to_graph[e1], nodes_to_graph[e2])
        E = Es[nodes_to_graph[e1]]  
        L = []
        if e1 in E:
            L = E[e1]
        L.append(e2)
        E[e1] = L
        #Es[nodes_to_graph[e1]] = E
    return Es



def read_graphs(folderpath, nr_graphs):
    classes = []
    Es = [{} for _ in range(nr_graphs)]
    Vs = [{} for _ in range(nr_graphs)]
    q = 1
    for i in range(nr_graphs):
        idx = q*i
        f = open(folderpath + str(idx) + '.txt', 'r')
        V = Vs[idx]
        E = Es[idx]
        for line in f:
            line_split = line.split('|')
            if len(line_split) < 2:
                c = int(line_split[0])
                classes.append(c)
                continue
            #print('line', line)
            edges_str = line_split[0]
            labels_str = line_split[1]
            edges_split = edges_str.split(',')
            labels_split = labels_str.split('::')
            u = int(edges_split[0])
            v = int(edges_split[1])
            if u > 100000 and len(labels_split[0].split(',')) > 1:
                continue
            if v > 100000 and len(labels_split[1].split(',')) > 1:
                continue
            if u not in V:
                V[u] = labels_split[0].strip()
            if v not in V:
                V[v] = labels_split[1].strip()
            E_u = []
            E_v = []
            if u in E:
                E_u = E[u]
            if v in E:
                E_v = E[v]
            E_u.append(v)
            E_v.append(u)
            E[u] = E_u
            E[v] = E_v
#        for u in E:
#            np.random.seed(u*i)
#            E[u] = np.random.permutation(E[u]).tolist()
        
    return Vs, Es, classes
        

def read_my_format(folderpath, nr_graphs, ratio):
    female_nr = int(ratio*nr_graphs)
    male_nr = int((1-ratio)*nr_graphs)
    cnt_m = 0
    cnt_f = 0
    classes = []
    Es = [{} for _ in range(nr_graphs)]
    Vs = [{} for _ in range(nr_graphs)]
    cnt_i = 0
    q = 1
    for i in range(2*q*int(nr_graphs/ratio)):
        if cnt_m == male_nr and cnt_f == female_nr:
            break
        #print(i)
        idx = q*i
        f = open(folderpath + str(idx) + '.txt', 'r')
        V = Vs[cnt_i]
        E = Es[cnt_i]
        for line in f:
            line_split = line.split('|')
            if len(line_split) < 2:
                c = int(line_split[0])
                if c == 0 and cnt_m < male_nr: 
                    classes.append(c)
                    cnt_m += 1
                    cnt_i += 1
                elif c == 1 and cnt_f < female_nr: 
                    classes.append(c)
                    cnt_f += 1
                    cnt_i += 1 
                else:
                    Vs[cnt_i] = {}
                    Es[cnt_i] = {}
                continue
            #print('line', line)
            edges_str = line_split[0]
            labels_str = line_split[1]
            edges_split = edges_str.split(',')
            labels_split = labels_str.split('::')
            u = int(edges_split[0])
            v = int(edges_split[1])
            if u > 100000 and len(labels_split[0].split(',')) > 1:
                continue
            if v > 100000 and len(labels_split[1].split(',')) > 1:
                continue
            if u not in V:
                V[u] = labels_split[0].strip()
            if v not in V:
                V[v] = labels_split[1].strip()
            E_u = []
            E_v = []
            if u in E:
                E_u = E[u]
            if v in E:
                E_v = E[v]
            E_u.append(v)
            E_v.append(u)
            E[u] = E_u
            E[v] = E_v
#        for u in E:
#            np.random.seed(u*i)
#            E[u] = np.random.permutation(E[u]).tolist()
        #E[v] = np.random.permutation(E_v).tolist()
    return Vs, Es, classes


def read_dh_format(folderpath, nr_graphs):
    classes = []
    Es = [{} for _ in range(nr_graphs)]
    Vs = [{} for _ in range(nr_graphs)]
    q = 1
    for i in range(nr_graphs):
        idx = q*i
        f = open(folderpath + str(idx) + '.txt', 'r')
        V = Vs[idx]
        E = Es[idx]
        for line in f:
            line_split = line.split('|')
            if len(line_split) < 2:
                c = int(line_split[0])
                classes.append(c)
                continue
            #print('line', line)
            edges_str = line_split[0]
            labels_str = line_split[1]
            edges_split = edges_str.split(',')
            labels_split = labels_str.split('::')
            u = int(edges_split[0])
            v = int(edges_split[1])
            if u > 100000 and len(labels_split[0].split(',')) > 1:
                continue
            if v > 100000 and len(labels_split[1].split(',')) > 1:
                continue
            if u not in V:
                V[u] = labels_split[0].strip()
            if v not in V:
                V[v] = labels_split[1].strip()
            E_u = []
            E_v = []
            if u in E:
                E_u = E[u]
            if v in E:
                E_v = E[v]
            E_u.append(v)
            E_v.append(u)
            E[u] = E_u
            E[v] = E_v
#        for u in E:
#            np.random.seed(u*i)
#            E[u] = np.random.permutation(E[u]).tolist()
    return Vs, Es, classes

        

def write_vectors_to_file(vectors, classes, filepath):
    #print(filepath)
    f = open(filepath, 'w')
    for i in range(len(vectors)):
        v = vectors[i]
        for item in v:
            f.write(str(item) + ' ')
        f.write('\n')
        f.write(str(classes[i]))
        f.write('\n')
    f.close()