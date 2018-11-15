"""
This is collection of functions that read a collection of graphs, generate neighborhood strings for each node,
and then generate/sketch explicit feature maps for the polynomial kernel for the k-gram frequency vector.
The reader is strongly advised to read the description the KONG paper: https://arxiv.org/pdf/1805.10014.pdf
"""

import read_write_utilities 
import tensorsketch
from count_sketch import CountSketch
import time
import numpy as np
import os
import platform
import copy


def normalize_vector(v):
    norm2 = np.linalg.norm(np.array(v))
    if norm2 == 0:
        norm2 = 1
    return [x/norm2 for x in v]

def normalize_vectors(vectors):
    norm_vectors = []
    for v in vectors:
        norm_vectors.append(normalize_vector(v))
    return norm_vectors


def order_nodes_by_label(Es, Vs):
    """
    a lexicographic ordering of the neighbor nodes, like in Weisfeiler-Lehman kernels
    """
    for i in range(len(Es)):
        E = Es[i]
        V = Vs[i]
        for u in E:
            labels_u = []
            for v in E[u]:
                labels_u.append((V[u], u))
            labels_u = sorted(labels_u)
            E[u] = [v for (_,v) in labels_u]
            
def order_nodes_by_degree(Es):
    """
    ordering the neighborhood nodes by their degree
    """
    for i in range(len(Es)):
        E = Es[i]
        for u in E:
            degrees_u = []
            for v in E[u]:
                if v in E:
                    deg_v = len(E[v])
                degrees_u.append((deg_v, v))
            degrees_u = sorted(degrees_u)
            E[u] = [v for (_,v) in degrees_u]
            

def sketch_polynomial_feature_map(label_vector, cs, cosine):
    """
    cs is a count-sketch data structure
    sketch a vector containing the label distribution 
    """
    if cosine:
        norm2 = np.linalg.norm(np.array(label_vector))
        if norm2 == 0:
            norm2 = 1
        label_vector = [x/norm2 for x in label_vector]
    cs.add_vector(label_vector)

def get_frequency_distribution(V):
    """
    compute the label frequency
    """
    Fr = {}
    for _,label in V.items():
         Fr[label] = Fr.setdefault(label, 0) + 1
    return Fr

def add_to_frequency_distribution(Fr, label):
    """
    add new label to the frequency distribution
    """
    Fr[label] = Fr.setdefault(label, 0) + 1
    return Fr

def BFS(v, V, E):
    """
    breadth-first search neighborhood traversal
    """
    N_v = []
    if v in E:
        N_v = E[v]
    new_label = []
    for u in N_v:
        u_label = V[u]
        new_label.append(u_label)
    return ','.join(str(e) for e in new_label)

#use this function for edge labeled data
def BFS_edge_label(v, V, E):
    """
    breadt-first search where both nodes and edges are labeled 
    """
    N_v = []
    if v in E:
        N_v = E[v]
    new_label = [V[v]]
    for (u,edge_label) in N_v:
        u_label = V[u]
        new_label.append(edge_label)
        new_label.append(u_label)
    return ','.join(str(e) for e in new_label)

def generate_feature_maps(V, E, h):
    """
    compute the label frequency of a graph by doing breadth-first search up to depth h
    """
    V_all = [copy.deepcopy(V) for _ in range(h+1)]
    V_all[0] = V
    feature_maps = [{} for _ in range(h+1)]
    feature_maps[0] = get_frequency_distribution(V)
    for i in range(h):
        for v in V_all[i]:
            if v not in E:
                continue
            label_v = BFS(v, V_all[i], E)
            V_all[i+1][v] = label_v
            add_to_frequency_distribution(feature_maps[i+1], label_v)
    return feature_maps



def feature_map_to_vector(feature_map, dirac_map):
    """
    convert a feature map to a vector. 
    dirac_map stores all labels in all graphs
    """
    label_vector = [0 for _ in range(len(dirac_map))]
    for label, val in feature_map.items():
        if label not in dirac_map:
            dirac_map[label] = len(dirac_map) 
        label_idx = dirac_map[label]
        if label_idx >= len(label_vector):
            label_vector.extend([0]*(label_idx - len(label_vector) + 1))
        label_vector[label_idx] += val 
    return label_vector


def feature_map_to_k_gram_vector(feature_map, labels_map, k):
    """
    convert a feature map to a vector by generating the k-grams from each feature/string
    """
    label_vector = [0 for _ in range(len(labels_map))]
    for label, val in feature_map.items():
        label_split = str.split(label, ',')
        for s in range(1, k+1):
            for i in range(len(label)-s):
                sublist = label_split[i:i+s]
                if len(sublist) < s:
                    continue
                l = ','.join(sub for sub in sublist)
                if l not in labels_map:
                    labels_map[l] = len(labels_map) 
                label_idx = labels_map[l]
                if label_idx >= len(label_vector):
                    label_vector.extend([0]*(label_idx - len(label_vector) + 1))
                label_vector[label_idx] += val 
    return label_vector

def compute_label_distribution(Vs):
    """
    write the label distribution of all graphs to vectors, such v[i] = cnt denotes that the i-th label appears cnt times
    """
    label_map = {}
    for i in range(len(Vs)):
        V = Vs[i]
        for _,label in V.items():
            if label not in label_map:
                label_map[label] = len(label_map)
    vectors = [[0 for _ in range(len(label_map))] for _ in range(len(Vs))]
    for i in range(len(Vs)):
        V = Vs[i]
        for _,label in V.items():
            vectors[i][label_map[label]] += 1        
    return vectors
        

def graph2map(Vs, Es, nr_graphs, h, k, table_size, random_files, nr_tables=1, max_p=2, dirac = False):
    """
    for a collection of graphs generate feature maps by traversing local neighborhoods, generating strings and sketching
    the k-gram ferquency distribution for 
    h: the depth at which neighborhood strings are generated
    k: the k in k-grams
    table_size: the count-sketch hashtable size
    random_files: needed for count-sketch initialization
    nr_tables: count-sketch parameter
    max_p: the maximum polynomial degree for the poly-kernel
    """
    print('Count sketch data structures initialization')
    cs = CountSketch(table_size, nr_tables*max_p, random_files)
    cs_cosine = CountSketch(table_size, nr_tables*max_p, random_files)
    
    print('Process graphs')
    vectors = [[] for _ in range(2*h)]
    vectors_cosine = [[] for _ in range(2*h)]
    dirac_vectors = []
    labels_maps = [{} for _ in range(h)]
    dirac_map = {}
    print(len(Vs), len(Es), nr_graphs)
    for i in range(nr_graphs):
        if i%500 == 0:
             print('i = ', i)
             print('number of features', len(labels_maps[0]))
        V,E = Vs[i], Es[i]
        feature_maps = generate_feature_maps(V, E, h)
        if dirac:
            dirac_vector = feature_map_to_vector(feature_maps[1], dirac_map)
            dirac_vectors.append(dirac_vector)
        for j in range(1, h+1):
            label_vector = feature_map_to_k_gram_vector(feature_maps[j], labels_maps[j-1], k)
            vectors[2*(j-1)].append(label_vector)
            vectors_cosine[2*(j-1)].append(normalize_vector(label_vector))
            
            cs.clear()
            cs_cosine.clear()
            sketch_polynomial_feature_map(label_vector, cs, False)
            sketch_polynomial_feature_map(label_vector, cs_cosine, True)
            vectors[2*(j-1) + 1].append(tensorsketch.compute_tensorsketch_from_cs(cs, max_p, nr_tables))
            vectors_cosine[2*(j-1) + 1].append(tensorsketch.compute_tensorsketch_from_cs(cs_cosine, max_p, nr_tables))

    for j in range(h):    
        maxlen_j = len(vectors[2*j][nr_graphs-1])
        print('total number of features', maxlen_j)
        for l in range(len(vectors[2*j])):
            vectors[2*j][l] += [0]*(maxlen_j-len(vectors[2*j][l]))
            vectors_cosine[2*j][l] += [0]*(maxlen_j-len(vectors_cosine[2*j][l]))
    if dirac:
        maxlen_dirac = len(dirac_vectors[nr_graphs-1])
        for l in range(len(dirac_vectors)):
            dirac_vectors[l] += [0]*(maxlen_dirac-len(dirac_vectors[l]))
    return vectors, vectors_cosine, dirac_vectors


if __name__ == "__main__":
    print('Graph string kernels')
    dirname = os.path.dirname(__file__)
    
    filenames = ['../Desktop/data/dunnhumby_graphs/', '../Desktop/data/MovieLens/', '../Desktop/data/kdd_graphs/']
    random_files = ['../data/random/']
    
    os_idx = 1
    if platform.system() == 'Windows':
        os_idx = 0
    

    filename = filenames[1 + os_idx]    
    h = 1
    k = 2
    max_p = 2
    table_size = 500
    max_value = 200000
    nr_tables = 1
    nr_graphs_per_class = 100
    dirac = False
    
    cs = CountSketch(table_size, nr_tables*max_p, random_files[os_idx], max_value)
    cs_cosine = CountSketch(table_size, nr_tables*max_p, random_files[os_idx], max_value)
    
    print('Count sketch data structures initialized')
    
    print(dirname, filename)
    ratio = 0.5
    Vs, Es, classes = read_write_utilities.read_my_format(filename, 3400, ratio)
    
    
    start = time.time()
    vectors, vectors_cosine, dirac_vectors = graph2map(Vs, Es, len(classes), h, k, table_size, nr_tables, max_p, dirac)
    print('elapsed time ', time.time()-start)
    
    
    outputpaths = ['../Desktop/data/ml1m_features']

    output = outputpaths[4 + os_idx] + '/'#'_perm/'
    print('length classes ', len(classes))
    print('len vectors', len(vectors))
    
    for j in range(h):
        read_write_utilities.write_vectors_to_file(vectors[2*j], classes, output + str(j+1) + '_1_' + str(k) + '.txt')
        read_write_utilities.write_vectors_to_file(vectors[2*j + 1], classes, output + str(j+1) + '_' + str(max_p) + '_' + str(k) + '.txt')
        read_write_utilities.write_vectors_to_file(vectors_cosine[2*j], classes, output + str(j+1) + '_1_cosine_' + str(k) + '.txt')
        read_write_utilities.write_vectors_to_file(vectors_cosine[2*j + 1], classes, output + str(j+1) + '_' + str(max_p) +  '_cosine_' + str(k) + '_' + str(table_size) + '.txt')
    if dirac:
        print('dimensionality of dirac', len(dirac_vectors[0]))
        dirac_vectors = normalize_vectors(dirac_vectors)
        read_write_utilities.write_vectors_to_file(dirac_vectors, classes, output + '_dirac.txt')
    