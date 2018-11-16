#!/usr/bin/env python

"""
  The following Count-Sketch implementation is optimized to work with truly random numbers for vectors with nonnegative integers in a small range. For details on Count-Sketch see http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/papers/Frequency-count/FrequentStream.pdf
  
Please contact Konstantin Kutzkov (kutzkov@gmail.com) if you have any questions.  
"""

import math
import struct
import time
import array
import operator
import numpy as np
from os import listdir
from os.path import isfile, join
from binary_stream import BinaryStream


class CountSketch(object):
    
    def __init__(self, m, nr, rnd_path,  max_value = 200000):
        """ `m` is the size of the hash table, larger m implies smaller
        error.
            `nr` is the number of different hash tables (sketches)
            `rnd_path` is the path to a file with truly random numbers, used for the hash function
            `max_value` is the maximum possible value of an item (needed for hash function implementation)
        """
        if not m:
            raise ValueError("Table size (m)  must be non-zero")
        self.m = m #size of hash table
        self.max_value = max_value
            
        self.nr = nr #number of different hash tables
        self.number = 0 #how many times the item appears in the stream
        self.norm = 0 #the squared 2-norm of the sketched vector
        self.sum = 0 #the total sum of the sketched vector
        self.sketches = [[0 for _ in range(m)] for _ in range(nr)]
        self.hashtables = []
        self.signtables = []
        randomfiles = [f for f in listdir(rnd_path) if isfile(join(rnd_path, f))]
        data = []
        bytes_array = []
        for rnd_f in randomfiles:
            f = open(join(rnd_path, rnd_f), mode='rb')
            for _ in range(max_value):
                data.append(struct.unpack('Q', f.read(8))[0])
                bytes_array.append(struct.unpack('b', f.read(1))[0])
        for k in range(nr):
            table = []
            signs = []
            for i in range(self.max_value + 1):
                table.append(data[k*max_value + i])
                if bytes_array[k*max_value + i] > 0:
                    signs.append(1)
                else:
                    signs.append(-1)
            self.hashtables.append(table)
            self.signtables.append(signs)
        print('Count-Sketch data structure initialized')    
    
    
    def _hash(self, x, seed):
        """ the hash value of x """
        if x > self.max_value or not isinstance(x, int) or x < 0:
            raise ValueError('Input number is not valid ', x)
        h = self.hashtables[seed][x] % self.m
        return  h
            
            
    def _sign(self, item, seed):
        return self.signtables[seed][item]

    def clear(self):
        """ delete all values from the sketches"""
        self.sketches = [[0 for _ in range(self.m)] for _ in range(self.nr)]
    
    def get(self, i):    
        return self.table[i]
        
    def update(self, item, val):
        """add a new (item, val) pair to the sketch"""
        for k in range(self.nr):
            i = self._hash(item, k)
            self.sketches[k][i] += self._sign(item, k)*val
            
    
    def add_vector(self, x):
        """add a vector to the sketch such that the items are its coordinates"""
        for i in range(len(x)):
            self.update(i, x[i])
        
    def get_table(self, i):    
        """return the i-th sketch"""
        return self.sketches[i]
    
    
    def print_sketches(self):
        for k in range(self.nr):
            print('Sketch', k)
            s_k = self.sketches[k]
            for i in range(self.m):
                if s_k[i] != 0:
                    print(i, s_k[i])
            print('\n')
        print('\n')
        
    def print_number(self):
        print(self.number)    
            
    def inner_prod_estimation(self, other):
        """"estimate the inner product of the current sketch and another sketched vector"""
        res = 0
        for i in range(self.m):
            res += self.get(i)*other.get(i)
        return res
    
    def get_norm(self):
        return math.sqrt(self.norm)
    
    def cosine_estimate(self, other):
        return self.inner_prod_estimation(other)/(self.getNorm()*other.getNorm())
    
    def get_avg(self):
        return float(self.sum)/float(self.number)
    
    def pearson_estimate(self, other):
        ip = self.inner_prod_estimation(other)
        f1 = self.get_avg()*other.sum
        f2 = other.get_avg()*self.sum
        f3 = min(self.number, other.number)*self.get_avg()*other.get_avg()
        return ip + f1 + f2 + f3
    
def median_estimate(item, sketches):
    res = []
    for cs in sketches:
        res.append(cs.estimate_frequency(item))
    return sorted(res)[len(res)//2]

def count_naively(filename, sep):
    items = {}
    fp = open(filename, 'r')
    for line in fp.readlines():
        line = line.strip()
        line = line.strip('\n')
        split_line = line.split(sep)
        for i in split_line:
            items.setdefault(i, 0)
            items[i] += 1
    return items       
    
