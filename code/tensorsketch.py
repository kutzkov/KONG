from count_sketch import CountSketch
import numpy as np
import math

def componentwise_multiplication(tables):
    if len(tables) < 2:
        raise ValueError('At least two tables needed')
    prod = [1 for _ in range(len(tables[0]))]
    for i in range(len(tables)):
        for j in range(len(prod)):
            prod[j] *= tables[i][j]   
    return np.fft.ifft(prod)


def compute_tensorsketch_from_cs(count_sketch, p, k):
    tensorsketches = [] 
    for i in range(k):
        tables_fft_i = [[] for _ in range(p)]
        for j in range(p):
            table_j = count_sketch.get_table(i*p + j)
            tables_fft_i[j] = np.fft.fft(table_j)
        sketch_i = componentwise_multiplication(tables_fft_i)
        sketch_i = [val/math.sqrt(k) for val in sketch_i]
        tensorsketches.append(sketch_i)
    result = []
    for sketch in tensorsketches:
        for val in sketch:
            realval = np.real(val)
            if realval > 0.00001:
                result.append(realval)
            else:
                result.append(0)
    return result

def compute_tensorsketch(count_sketch, v, p, k):
    count_sketch.add_vector(v)
    
    tensorsketches = [] 
    for i in range(k):
        tables_fft_i = [[] for _ in range(p)]
        for j in range(p):
            table_j = count_sketch.get_table(i*p + j)
            tables_fft_i[j] = np.fft.fft(table_j)
        sketch_i = componentwise_multiplication(tables_fft_i)
        sketch_i = [val/math.sqrt(k) for val in sketch_i]
        tensorsketches.append(sketch_i)
    count_sketch.clear()
    return [np.real(val) for sketch in tensorsketches for val in sketch]

if __name__=="__main__":
    print('Tensorsketch')
    random_files = ['<path to random files>', '']
    table_size = 300
    v1 = np.array([10, 2, 3, 1100, 3, 28, 300, 12, 3, 21, 11, 20,18,16,31,300], dtype=np.uint64)
    v2 =  np.array([10, 29, 3, 1001, 3, 28, 109, 12, 13, 21, 110, 20,108,16,301,30], dtype=np.uint64)
    print(len(v1))
    print(len(v2))
    p = 2
    k = 1
    cs = CountSketch(table_size, k*p, random_files[0], 1000)
    tensorsketches1 = compute_tensorsketch(cs, v1, p, k)
    tensorsketches2 = compute_tensorsketch(cs, v2, p, k)
    
    
    print(np.dot(v1, v2)**p)
    print(np.real(np.dot(tensorsketches1, tensorsketches2)))
    
