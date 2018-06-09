import pandas as pd
import numpy as np

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_data(filename):
    df = pd.read_csv(filename)
    print(df.columns)
    customers = df.groupby(['CUST_CODE', 'CUST_PRICE_SENSITIVITY', 'CUST_LIFESTAGE']).groups.keys()
    #print(len(customers))
    products = df.groupby(['PROD_CODE', 'PROD_CODE_40']).groups.keys()
    #print(len(products))
    
    cust_map = {}
    for c in customers:
        if c[0] in cust_map:
            #print('deleted')
            del cust_map[c[0]]
            continue
        cust_map[c[0]] = (c[1], c[2])
    print(len(cust_map))
    
    prod_map = {}
    for p in products:
        if p[0] in prod_map:
            #print('deleted')
            del prod_map[p[0]]
            continue
        prod_map[p[0]] = p[1]
    print(len(prod_map))
    
    lines = []
    for index, row in df.iterrows():
        if is_number(row['SHOP_WEEK']):
            idx = (int(row['SHOP_WEEK']) - 200626)*192 + int(row['SHOP_WEEKDAY'])*24 + int(row['SHOP_HOUR'])
        
            lines.append((idx, str(row['CUST_CODE']), str(row['PROD_CODE'])))# [row['CUST_CODE'], row['PROD_CODE']]))
 
    lines = sorted(lines)
    print(len(cust_map))
    print(len(lines))
    
    E = {}
    for l in lines:
        curr_time = l[0]
        u = l[1]
        v = l[2]
        if u == 'nan' or v == 'nan':
            continue
        E_u = []
        if u in E:
            E_u = E[u]
        E_u.append((curr_time, v))
        E[u] = E_u
        E_v = []
        if v in E:
            E_v = E[v]
        E_v.append((curr_time, u))
        E[v] = E_v
        
    for u in E:
        E_u = sorted(E[u])
        E_u = [v for ts,v in E_u]
        E[u] = E_u
            
    return E, cust_map, prod_map


def write_to_graphs(E, customers, products, outpath):
    lifestage_idx = {}
    cust_idx = {}
    prod_idx = {}
    for u in E:
        if u in products:
            if u not in prod_idx:
                prod_idx[u] = len(prod_idx) 
        if u in customers:
            if u not in cust_idx:
                cust_idx[u] = len(cust_idx) + 10000
    
    cnt_files = 0
    f = open(outpath + '0.txt', 'w')
    for u in E:
        if u in customers:
            vals_u = customers[u]
            ls = vals_u[1]
            if ls not in lifestage_idx:
                lifestage_idx[ls] = len(lifestage_idx)
            ls_idx = lifestage_idx[ls]
            for v in E[u]:
                f.write(str(cust_idx[u]) + ',' + str(prod_idx[v]) + '|' + str(vals_u[0]) + '::' + str(products[v]) + '\n')
                for w in E[v]:
                    #print('w', w)
                    vals_w = customers[w]
                    f.write(str(prod_idx[v]) + ',' + str(cust_idx[w]) + '|' + str(products[v]) + '::' + str(vals_w[0]) + '\n')
            f.write(str(ls_idx))
            f.close()
            cnt_files += 1
            f = open(outpath + str(cnt_files) + '.txt', 'w')
            
def to_graphs(filename, outputname):
    E, cust_map, prod_map = read_data(filename)
    write_to_graphs(E, cust_map, prod_map, outputname)
    
    
if __name__== '__main__':
    print('Dunnhumby')
    
    filenames = ['C:/Users/KUTZKOV/Desktop/data/dunnhumby/', 'C:/Users/KUTZKOV/Desktop/data/dunnhumby_graphs/']
    E, cust_map, prod_map = read_data(filenames[0] + 'merged.csv')
    write_to_graphs(E, cust_map, prod_map, filenames[1])