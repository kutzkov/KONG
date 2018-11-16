#!/usr/bin/env python

"""
functions that read files in the MovieLens format https://grouplens.org/datasets/movielens/
and create graphs as explained in the paper: https://arxiv.org/pdf/1805.10014.pdf

Please contact Konstantin Kutzkov (kutzkov@gmail.com) if you have any questions.
"""

import platform


#users: id, gender, age, occupation
def read_users(filename):
    f = open(filename, 'r', encoding = "ISO-8859-1")
    users = {}
    occupations = set()
    for line in f:
        line = line.strip()
        line_split = line.split('::')
        if len(line_split) != 5:
            print('line', line)
            continue
        users[int(line_split[0])] = [line_split[1], int(line_split[2]), line_split[3]]
        occupations.add(line_split[3])
    return users, occupations
        
   
    

def read_items(filename):
    """read movies in the fomat movie: id::name::genre|genre|...|genre
       and store them in dictionary iff the movie has a single genre
    """
    f = open(filename, 'r', encoding = "ISO-8859-1")
    items = {}
    for line in f:
        line = line.strip()
        line_split = line.split('::')
        genre = line_split[2]
        genre_split = genre.split('|')
        label = ''
        if len(genre_split) == 1:
            items[int(line_split[0]) + 100000] = genre.strip()
    return items

#user, movie, rating, timestamp
def read_data(filename):
    """"read user ratings in the format user::movie::rating::timestamp
        and create labeled graphs as described in the paper"""
    f = open(filename, 'r', encoding = "ISO-8859-1")
    E = {}
    cnt = 0
    list_of_lines = []
    for line in f:
        line = line.strip()
        cnt += 1
        line_split = line.split('::')
        list_of_lines.append((int(line_split[3]), [line_split[0], line_split[1], line_split[2]]))
    list_of_lines = sorted(list_of_lines)
    for _,vals in list_of_lines:
        u = int(vals[0])
        v = 100000 + int(vals[1])
        nbrs_u = []
        nbrs_v = []
        if u in E:
            nbrs_u = E[u]
        if v in E:
            nbrs_v = E[v]
        nbrs_u.append(v)
        nbrs_v.append(u)
        E[u] = nbrs_u
        E[v] = nbrs_v    
    return E
    

def create_age_graphs(E, users, items, nr_nodes, outpath):
    classes = []
    cnt_files = 0
    q = 15
    f = open(outpath + '0.txt', 'w')
    for u in E:
        if cnt_files % 10 == 0:
            print(cnt_files)
        if u in users:
            vals_u = users[u]
            print(vals_u)
            age_group = str(users[u][0]//q)
            for v in E[u]:
                    f.write(str(u) + ',' + str(v) + '|' + str(users[u][1]) + '-' + items[v] + '\n')
                    for w in E[v]:
                        #print(w)
                        f.write(str(v) + ',' + str(w) + '|' + items[v] +  '-' + str(users[w][1]) + '\n')
    
def create_graphs(E, users, items, outpath, nr=10000):
    cnt_files = 0
    q = 15
    f = open(outpath + '0.txt', 'w')
    cnt_female = 0
    for u in E:
        if cnt_files % 100 == 0:
            print('graphs generated so far', cnt_files)
        if u in users:
            vals_u = users[u]
            gender_u = vals_u[0]
            if gender_u == 'M':
                for v in E[u]:
                    if v in items:
                        f.write(str(u) + ',' + str(v) + '|' + str(users[u][1]//q) + '::' + items[v] + '\n')
                        for w in E[v]:
                            f.write(str(v) + ',' + str(w) + '|' + items[v] +  '::' + str(users[w][1]//q) + '\n')
                f.write('0')
                f.close()
                cnt_files += 1
                if cnt_files > nr:
                    break
                f = open(outpath + str(cnt_files) + '.txt', 'w')
            else:
                cnt_female += 1
                for v in E[u]:
                    if v in items:
                        f.write(str(u) + ',' + str(v) + '|' + str(users[u][1]//q) + '::' + items[v] + '\n')
                        for w in E[v]:
                            f.write(str(v) + ',' + str(w) + '|' + items[v] + '::' + str(users[w][1]//q) + '\n')
                f.write('1')
                f.close()
                cnt_files += 1
                if cnt_files > nr:
                    break
                f = open(outpath + str(cnt_files) + '.txt', 'w')
          
        
if __name__ == '__main__':
    dirname = ['data/ml-1m/', 'data/Graphs/MovieLens/']
    users, occupations = read_users(dirname[0] + 'users.dat')
    items = read_items(dirname[0] + 'movies.dat')
    E = read_data(dirname[0] + 'ratings.dat')
    create_graphs(E, users, items, dirname[1])