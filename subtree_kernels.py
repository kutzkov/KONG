import numpy as np
import math
import matplotlib.pylab as plt
#from os import system
from sklearn.metrics.pairwise import cosine_similarity
import os
import platform
import read_write_utilities 
import tensorsketch
from count_sketch import CountSketch


def get_label_tensor(label, p):
    i = 0
    tensor_new = {'':1}
    label_split = str.split(label, ',')
    Fr = {}
    for c in label_split:
        Fr[c] = Fr.setdefault(c, 0) + 1
    for i in range(p):
        tensor_old = tensor_new
        tensor_new = {}
        for t in tensor_old:
            for c in Fr:
                new_t = c + t
                tensor_new[new_t] = Fr[c]*tensor_old[t]
    return tensor_new
                

def get_frequency_distribution(V):
    Fr = {}
    for _,label in V.items():
         Fr[label] = Fr.setdefault(label, 0) + 1
    return Fr

def get_frequency_from_string(label):
    Fr = {}
    for c in label:
        Fr[c] = Fr.setdefault(c, 0) + 1
    return Fr

def add_to_frequency_distribution(Fr, label):
    Fr[label] = Fr.setdefault(label, 0) + 1
    return Fr

def add_frequency_vector(feature_map, Fr):
    for v, val_v in Fr.items():
        cnt_v = 0
        if v in feature_map:
            cnt_v = feature_map[v]
        feature_map[v] = cnt_v + val_v



def BFS(v, V, E):
    N_v = []
    if v in E:
        N_v = E[v]
    new_label = []#V[v]
    for u in N_v:
        u_label = V[u]
        #new_label += u_label
        #print('u label', u_label)
        new_label.append(u_label)
    new_label = sorted(new_label)
    new_label.insert(0, V[v])
    #print('label', new_label)
    return ','.join(str(e) for e in new_label)#''.join(sorted(new_label))
    #print('new label', new_label)
    #return ''.join(sorted(new_label))  

def BFS_WL(v, V, E): 
    N_v = []
    if v in E:
        N_v = E[v]
    new_label = []
    for u in N_v:
        u_label = V[u]
        new_label.append(u_label)
    new_label = sorted(new_label)
    new_label.insert(0, V[v])
    return ''.join(str(e) for e in new_label)#''.join(sorted(new_label))

def inner_product(fr1, fr2):
    ip = 0
    for k in fr1:
        if k in fr2:
            ip += fr1[k]*fr2[k]
    return ip


def cosine(fr1, fr2):
    norm1 = math.sqrt(inner_product(fr1,fr1))
    norm2 = math.sqrt(inner_product(fr2,fr2))
    return inner_product(fr1, fr2)/(norm1*norm2)
        
def WL_label(v, V, E):
    N_v = E[v]
    label = []
    for u in N_v:
        label.append(V[u])
    return ''.join(sorted(label))


def generate_feature_maps_relabel(V, E, V_labels, k, h):
    V_all = [{} for _ in range(h+1)]
    cnt_unique_labels = len(V_labels.keys())
    relabel_feature_maps = [{} for _ in range(h+1)]
    relabel_feature_maps[0] = get_frequency_distribution(V)
    V_all[0] = V
    for i in range(k):
        #print('i = ', i)
        for v in V_all[i]:
            label_v = BFS(v, V_all[i], E)
            if label_v in V_labels:
                V_all[i+1][v] = V_labels[label_v]
            else:
                V_all[i+1][v] = cnt_unique_labels
                V_labels[label_v] = cnt_unique_labels
                cnt_unique_labels += 1
            #V_all[i+1][v] = label_v
            #print('adding', label_v, ' with new label ', V_labels[label_v])
            add_to_frequency_distribution(relabel_feature_maps[i+1], str(V_labels[label_v]))
            #print(relabel_feature_maps)
    for i in range(k, h):
        #print('i = ', i)
        for v in V_all[i]:
            label_v = BFS(v, V_all[i], E)
            V_all[i+1][v] = label_v
            add_to_frequency_distribution(relabel_feature_maps[i+1], label_v)
    return relabel_feature_maps
        
def generate_feature_maps(V, E, h):
    V_all = [{} for _ in range(h+1)]
    V_all[0] = V
    #print(V)
    feature_maps = [{} for _ in range(h+1)]
    feature_maps[0] = get_frequency_distribution(V)
    for i in range(h):
        #print(V_all[i])
        for v in V_all[i]:
            label_v = BFS(v, V_all[i], E)
            V_all[i+1][v] = label_v
            add_to_frequency_distribution(feature_maps[i+1], label_v)
    return feature_maps

def generate_WL_feature_maps(V, E, V_labels, h):
    V_all = [{} for _ in range(h+1)]
    V_all[0] = V
    #V_labels = {}
    cnt_unique_labels = len(V_labels.keys())
    WL_feature_maps = [{} for _ in range(h+1)]
    WL_feature_maps[0] = get_frequency_distribution(V)
    for i in range(h):
        for v in V_all[i]:
            label_v = BFS(v, V_all[i], E)
            if label_v in V_labels:
                V_all[i+1][v] = V_labels[label_v]
            else:
                V_all[i+1][v] = cnt_unique_labels
                V_labels[label_v] = cnt_unique_labels
                cnt_unique_labels += 1
            add_to_frequency_distribution(WL_feature_maps[i+1], V_labels[label_v])
    return WL_feature_maps


def feature_map_to_vector(feature_map, labels_map):
    label_vector = [0,0]
    for label, val in feature_map.items():
        label_split = str.split(label, ',')
        for l in label_split:
            if l not in labels_map:
                labels_map[l] = len(labels_map) 
            label_idx = labels_map[l]
            if label_idx >= len(label_vector):
                label_vector.extend([0]*(label_idx + 2))
            label_vector[label_idx] += val 
    return label_vector

def sketch_polynomial_feature_map(label_vector, cs, cosine):# p, table_size, nr_tables, hash_function_file, sign_function_file, max_val):
    if cosine:
        norm2 = np.linalg.norm(np.array(label_vector))
        label_vector = [x/norm2 for x in label_vector]
    cs.add_vector(label_vector)
        
    
def compute_polynomial_feature_map(feature_map, p):
    graph_map = {}
    cnt_total = 0
    for label,cnt in feature_map.items():
        #print(label, cnt)
        fm_p = get_label_tensor(label, p)
        #print('fm_p', fm_p)
        for pattern,cnt_p in fm_p.items():
            cnt_pattern = cnt*cnt_p
            if pattern in graph_map:
                cnt_pattern += graph_map[pattern]
            graph_map[pattern] = cnt_pattern
            cnt_total += cnt*cnt_p
    return graph_map, cnt_total
                
 
def get_vector_from_map(graph_map, index_map, cosine_sim):
    norm = 0
    for label,cnt in graph_map.items():
        norm += cnt*cnt
        if label in index_map:
            idx = index_map[label]
        else:
            idx = len(index_map.keys())
            index_map[label] = idx
    norm = math.sqrt(norm)
    if not cosine_sim:
        norm = 1
    v = [0 for _ in range(len(index_map.keys()))]
    for label,cnt in graph_map.items():
        if cnt/norm > 0:
            v[index_map[label]] = cnt/norm
    return v

def graph2WLmap(Vs, Es, nr_graphs, set_labels, h):
    WL_feature_maps = [[] for _ in range(h+1)]
    V_labels = {}
    cnt_labels = 0
    for l in set_labels:
        V_labels[l] = cnt_labels
        cnt_labels += 1
    for i in range(1,nr_graphs):
        if i%100 == 0:
             print(i)
        V,E = Vs[i], Es[i]
        maps = generate_WL_feature_maps(V, E,  V_labels, h)
        #print(len(maps))
        for k in range(len(maps)):
            WL_feature_maps[k].append(maps[k])
    return WL_feature_maps
 
def graph2map(Vs, Es, nr_graphs, set_labels, h, relabel, cs, cs_cosine, nr_tables, max_p = 4):
    index_maps = [{} for _ in range(h*max_p)]
    vectors = [[] for _ in range(h*max_p)]
    vectors_cosine = [[] for _ in range(h*max_p)]
    #vectors_sketched = [[] for _ in range(h*max_p)]
    V_labels = {}
    cnt_labels = 0
    labels_map = {}
    for l in set_labels:
        V_labels[l] = cnt_labels
        cnt_labels += 1
    print(len(Vs), len(Es), nr_graphs)
    for i in range(1,nr_graphs):
        if i%100 == 0:
             print(i)
             print(len(labels_map))
        V,E = Vs[i], Es[i]
        feature_maps = []
        if relabel:
            feature_maps = generate_feature_maps_relabel(V, E, V_labels, 1, h)
            #max_p = 2
        else:
            feature_maps = generate_feature_maps(V, E, h)
        #print(feature_maps)
        for k in range(1, h+1):
            cs.clear()
            cs_cosine.clear()
            label_vector = feature_map_to_vector(feature_maps[k], labels_map)
            sketch_polynomial_feature_map(label_vector, cs, False)
            sketch_polynomial_feature_map(label_vector, cs_cosine, True)
            #sketch_polynomial_feature_map(feature_maps[k], labels_map, cs, max_p, nr_tables)
            for p in range(1, max_p+1):
                #all_feature_maps[(k-1)*max_p + p-1].append(feature_maps[k])
                if p > 1 and len(label_vector) > 100:
                    vectors[(k-1)*max_p + p-1].append(tensorsketch.compute_tensorsketch_from_cs(cs, p, nr_tables))
                    #vectors[(k-1)*max_p + p-1].append([0])
                    vectors_cosine[(k-1)*max_p + p-1].append(tensorsketch.compute_tensorsketch_from_cs(cs_cosine, p, nr_tables))
                else:
                    graph_map, _ = compute_polynomial_feature_map(feature_maps[k], p)
                    vectors[(k-1)*max_p + p-1].append(get_vector_from_map(graph_map, index_maps[(k-1)*max_p + p-1], False))
                    vectors_cosine[(k-1)*max_p + p-1].append(get_vector_from_map(graph_map, index_maps[(k-1)*max_p + p-1], True))
    for j in range(len(vectors)):
        maxlen_j = len(vectors[j][nr_graphs-2])
        for l in range(len(vectors[j])):
            vectors[j][l] += [0]*(maxlen_j-len(vectors[j][l]))
            vectors_cosine[j][l] += [0]*(maxlen_j-len(vectors_cosine[j][l]))
    return vectors, vectors_cosine
       
def feature_map_per_graph(file_dir, filename, nr, h, k, p, read_edges, cosine_sim = False):
    filename_base = os.path.join(file_dir, filename)
    print(filename)
    index_map = {}
    vectors = []
    classes = []
    all_feature_maps = []
    for i in range(1,nr+1):
        if i%100 == 0:
            print(i)
        filename = filename_base + str(i) + '.graph'
        V,E,C = {}, {}, ''
        if read_edges:
            V,E,C = read_write_utilities.read_graph_edges(filename)
        else:
            V,E,C = read_write_utilities.read_graph_adj_list(filename)
        feature_maps = generate_feature_maps(V, E, h)
        #print('feature map k',feature_maps[k])
        all_feature_maps.append(feature_maps[k])
        #for k in range(h-1,h):
        graph_map, _ = compute_polynomial_feature_map(feature_maps[k], p)
        #print('graph map', graph_map)
        graph_vector = get_vector_from_map(graph_map, index_map, C, cosine_sim)
        #print(len(graph_vector))
        #print(C, graph_vector[:10])
        vectors.append(graph_vector)
        classes.append(C)
        #print(len(graph_vector))
    maxlen = len(vectors[nr-2])
    print('maxlen', maxlen)
    for i in range(len(vectors)):
        vectors[i] += [0]*(maxlen-len(vectors[i]))
    return vectors, classes, all_feature_maps
        
            
#def write_vectors_to_file(vectors, classes, filepath):
#    #print(filepath)
#    f = open(filepath, 'w')
#    for i in range(len(vectors)):
#        v = vectors[i]
#        for item in v:
#            f.write(str(item) + ' ')
#        f.write('\n')
#        f.write(classes[i])
#        f.write('\n')
#    f.close()
    
def WL_map_to_vector(feature_map, label_map):
    nr_of_labels = len(label_map.keys())
    vector = [0 for _ in range(nr_of_labels)]
    
    for label,cnt in feature_map.items():
        label_idx = nr_of_labels
        if label in label_map:
            label_idx = label_map[label]
            vector[label_idx] = cnt
        else:
            label_map[label] = label_idx
            nr_of_labels += 1
            vector.append(cnt)
    return vector

def write_WL_vectors_to_file(WL_feature_maps, k, classes, filepath):
    label_path = {}
    vectors = [[] for _ in range(len(classes))]
    for i in range(k):
        WL_feature_maps_i = WL_feature_maps[i]
        cnt_graphs = 0
        for fm in WL_feature_maps_i:
            v_G_i = WL_map_to_vector(fm, label_path)
            vectors[cnt_graphs].extend(v_G_i)
            cnt_graphs += 1
            #print(v)
        maxlen = len(vectors[-1])
        #print('WL maxlen', maxlen)
        for i in range(len(vectors)):
            vectors[i] += [0]*(maxlen-len(vectors[i]))
    read_write_utilities.write_vectors_to_file(vectors, classes, filepath)
    