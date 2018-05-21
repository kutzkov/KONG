import pickle
import os
import numpy as np


def count_edges_my_data(folderpath, nr):
    cnt = 0
    V = set()
    nr_nodes = 0
    for idx in range(nr):
        if not os.path.isfile(folderpath + str(idx) + '.txt'):
            break
        f = open(folderpath + str(idx) + '.txt', 'r')
        for line in f:
            
            line_split = line.split('|')
            if len(line_split) < 2:
                continue
            v_split = line_split[0].split(',')
            V.add(v_split[0])
            V.add(v_split[1])
            cnt += 1
        nr_nodes += len(V)
        V.clear()
        f.close()
    print('Nr of nodes in ', folderpath, ' is ', nr_nodes)
    print('Nr of edges in ', folderpath, ' is ', cnt)

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



def read_dh_graphs(folderpath, nr_graphs):
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


#++++++++++++++++++++++++++++++++++++++++++++++++++++
#Currently not needed but might be useful in future
#++++++++++++++++++++++++++++++++++++++++++++++++++++

def read_graph_edges(filename, sep = ','):
    f = open(filename, 'r')
    V = {}
    E = {}
    C = ''
    v_idx = 1
    read_class = False
    for line in f:
        if line[:2] == '#c':
            read_class = True
            continue
        if line[0] == '#':
            continue
        line_split = str.split(line, sep)
        if (len(line_split) < 1):
            continue
        elif len(line_split)==1:
            if read_class:
                C = line_split[0].strip()
            else:
                V[v_idx] = line_split[0].strip()
                v_idx += 1
        else:
            L = []
            e1 = int(line_split[0].strip())
            e2 = int(line_split[1].strip())
            if e1 in E:
                L = E[e1]
            L.append(e2)
            E[e1] = L
    return V,E,C
    
def read_graph_adj_list(filename, sep=','):
    f = open(filename, 'r')
    V = {}
    E = {}
    C = ''
    v_idx = 1
    v_idx_edges = 1
    read_class = False
    read_label = False
    #read_neighbors = False
    for line in f:
        if line[:2] == '#c':
            read_class = True
            continue
        if line[:2] == '#v':
            read_label = True
            continue
        if line[:2] == '#a':
            #read_neighbors = True
            read_label = False
            continue
        line_split = str.split(line, sep)
        if len(line_split)==1:
            if read_class:
                C = line_split[0].strip()
                read_class = False
                continue
            if read_label:
                V[v_idx] = line_split[0].strip()
                v_idx += 1
                continue
        L = []
        for i in range(len(line_split)):
            val = line_split[i].strip()
            if v_idx_edges in E:
                L = E[v_idx_edges]
            if len(val) > 0:
                L.append(int(val))
            E[v_idx_edges] = L
        v_idx_edges += 1
    return V,E,C


#########################################################
# KDD data readers and writers
#########################################################
    
def read_tsv_files(pathname, n):
    vals = [100*j + i for j in range(6) for i in range(n)]
    perm = np.random.permutation(vals)
    Vs = []
    Es = []
    classes = []
    nr_nodes = 0
    nr_edges = 0
    for p in perm:
        filename = pathname + str(p) + '.txt'
        V,E, nr = read_tsv_file(filename)
        nr_nodes += len(V)
        nr_edges += nr
        Vs.append(V)
        Es.append(E)
        classes.append(str(p//100))
    #print(Es)
    print('nr of nodes', nr_nodes)
    print('nr of edges', nr_edges)
    return Vs, Es, classes
            
def read_tsv_file(filename):
    f = open(filename, 'r')
    #cnt = 0
    #Vs = []
    #Es = []
    
    V_i = {}
    E_i = {}
    nr_edges = 0
    #current_graph = 0
    for line in f:
        nr_edges += 1
        #cnt += 1
        #if cnt % 1000000 == 0:
        #    print(cnt)
        line_split = str.split(line, '\t')
        #print(line)
        #graph_id = int(line_split[5])
        #if graph_id % 100 > 50:
        #    continue
#        if graph_id != current_graph:
#            Vs.append(V_i)
#            Es.append(E_i)
#            V_i.clear()
#            E_i.clear()
#            current_graph = graph_id
        u = int(line_split[0])
        label_u = line_split[1]
        v = int(line_split[2])
        label_v = line_split[3]
        if u in V_i:
            if V_i[u] != label_u:
                print('Error, different labels of vertex ', u)
        else:
            V_i[u] = label_u
        if v in V_i:
            if V_i[v] != label_v:
                print('Error, different labels of vertex ', v)
        else:
            V_i[v] = label_v
        edge_labels_u = []
        if u in E_i:
            edge_labels_u = E_i[u]
        edge_labels_u.append((v, line_split[4]))
        E_i[u] = edge_labels_u  
#    k = 0    
#    for u in E_i:
#        k += 1
#        np.random.seed(u*k)
#        indices = np.random.permutation(len(E_i[u]))
#        new_nbrs = []
#        for idx in indices:
#            new_nbrs.append(E_i[u][idx])
#        E_i[u] = new_nbrs#np.random.permutation(E_i[u])
    return V_i, E_i, nr_edges
#        if cnt > 10000:
#            break
#    Vs.append(V_i)
#    Es.append(E_i)
#    print('nr of edges', cnt)
#    print(len(Vs), len(Es))
#    print(len(Vs[0]), len(Es[0]))
    
    
def write_tsv_graphs_to_files(filename, outputpath):
    f = open(filename, 'r')
    cnt = 0
    current_graph = 0
    outfilename = outputpath + str(current_graph) + '.txt'
    fout = open(outfilename, 'w')
    for line in f:
        cnt += 1
        if cnt % 1000000 == 0:
            print(cnt)
        line_split = str.split(line, '\t')
        graph_id = int(line_split[5])
        if graph_id != current_graph:
            current_graph = graph_id
            fout.close()
            outfilename = outputpath + str(current_graph) + '.txt'
            fout = open(outfilename, 'w')
        fout.write(line)
        
if __name__ == "__main__":
    print('Read write utilities')
    folderpath = ['/data/dunnhumby_graphs/']
    
    read_tsv_files(folderpath[0], 100)