from pulp import *
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import pprint
import networkx as nx
import numpy as np, pandas as pd
from more_itertools import iterate, take
from student_utils import *


###################################################################################################
def tsp(nodes, list_of_homes, dist=None):
    """
    巡回セールスマン問題
    入力
        nodes: 点(dist未指定時は、座標)のリスト
        dist: (i,j)をキー、距離を値とした辞書
    出力
        距離と点番号リスト
    """

    n = len(nodes)

    if not dist:
        dist = {(i, j): np.linalg.norm(np.subtract(nodes[i], nodes[j]))
                for i in range(n) for j in range(i + 1, n)}
        dist.update({(j, i): d for (i, j), d in dist.items()})

    # data farme containing distances from node i to node j
    print(dist)
    a = pd.DataFrame([(nodes[i], i, j, dist[(i, j)])
                      for i in range(n) for j in range(n) if i != j], columns=['Name', 'NodeI', 'NodeJ', 'Dist'])

    m = LpProblem()

    # creates x_ij for every edge in the graph
    a['VarIJ'] = [LpVariable('x%d' % i, cat=LpBinary) for i in a.index]

    a['VarJI'] = a.sort_values(['NodeJ', 'NodeI']).VarIJ.values

    #
    u = [0] + [LpVariable('y%d' % i, lowBound=0) for i in range(n - 1)]
    print(a)
    # gives the total distance,
    m += lpDot(a.Dist, a.VarIJ)
    print(list_of_homes)
    for _, v in a.groupby('NodeI'):
        print(v)
        print(v.iloc[0, :].NodeI)
        print('*****')
        if v.iloc[0, :].Name in list_of_homes:
            print('*&&&&', v.iloc[0, :].Name)
            m += lpSum(v.VarIJ) == 1  # constraint for one edge exiting each vertex
            m += lpSum(v.VarJI) == 1  # constraint for one edge entering each vertex

    for  _, (name, i, j, _, vij, vji) in a.query('NodeI!=0 & NodeJ!=0').iterrows():
        m += u[i] + 1 - (n - 1) * (1 - vij) + (n - 3) * vji <= u[j]  # 持ち上げポテンシャル制約(MTZ)

    for _, (name, _, j, _, v0j, vj0) in a.query('NodeI==0').iterrows():
        m += 1 + (1 - v0j) + (n - 3) * vj0 <= u[j]  # lower bound constraints
    for _, (name, i, _, _, vi0, v0i) in a.query('NodeJ==0').iterrows():
        m += u[i] <= (n - 1) - (1 - vi0) - (n - 3) * v0i  # upper bound constraints
    m.solve()
    a['ValIJ'] = a.VarIJ.apply(value)
    dc = dict(a[a.ValIJ > 0.5][['NodeI', 'NodeJ']].values)
    return value(m.objective), list(take(n, iterate(lambda k:dc[k], 0)))

def greedyAllPairs(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    #print(adjacency_matrix)
    G = nx.Graph(incoming_graph_data=adjacency_matrix, cutoff=1000)
    predecessors, distances = nx.floyd_warshall_predecessor_and_distance(G)
    print(list(distances.items())[0][1][0])
    homeList = []
    def find_closest_home_to_location(location):
        distance = float('inf')
        closest_home = list_of_homes[0]
        for h in list_of_homes:
            home_idx = list_of_locations.index(h)
            location_idx = list_of_locations.index(location)
            new_dist = list(distances.items())[home_idx][1][location_idx]

            if new_dist < distance:
                distance = new_dist
                closest_home = h
        return closest_home, distance

    current = starting_car_location
    total_path = [list_of_locations.index(starting_car_location)]
    dropoff_mapping = {}

    for _ in range(len(list_of_homes)):
        closest, distance = find_closest_home_to_location(current)
        homeList.append(closest)

        curr_idx = list_of_locations.index(current)
        next_idx = list_of_locations.index(closest)
        path_to_closest = nx.reconstruct_path(curr_idx, next_idx, predecessors)
        dropoff_mapping[next_idx] = [next_idx]
        total_path.extend(path_to_closest[1:])

        list_of_homes.remove(closest)
        current = closest

    start_idx = list_of_locations.index(starting_car_location)
    path_to_start = nx.reconstruct_path(next_idx, start_idx, predecessors)
    total_path.extend(path_to_start[1:])
    print(dropoff_mapping)
    return total_path, dropoff_mapping, homeList

def dropOffPath(list_of_locations, list_of_homes, drop_to_home, starting_car_location, adjacency_matrix):
    #print(adjacency_matrix)
    G = nx.Graph(incoming_graph_data=adjacency_matrix, cutoff=1000)
    predecessors, distances = nx.floyd_warshall_predecessor_and_distance(G)
    print(list(distances.items())[0][1][0])
    homeList = []
    def find_closest_home_to_location(location):
        distance = float('inf')
        closest_home = list_of_homes[0]
        for h in list_of_homes:
            home_idx = list_of_locations.index(h)
            location_idx = list_of_locations.index(location)
            new_dist = list(distances.items())[home_idx][1][location_idx]

            if new_dist < distance:
                distance = new_dist
                closest_home = h
        return closest_home, distance

    current = starting_car_location
    total_path = [list_of_locations.index(starting_car_location)]
    dropoff_mapping = {}

    for _ in range(len(list_of_homes)):
        closest, distance = find_closest_home_to_location(current)
        homeList.append(closest)

        curr_idx = list_of_locations.index(current)
        next_idx = list_of_locations.index(closest)
        path_to_closest = nx.reconstruct_path(curr_idx, next_idx, predecessors)
        dropoff_mapping[next_idx] = [drop_to_home[next_idx]]
        total_path.extend(path_to_closest[1:])

        list_of_homes.remove(closest)
        current = closest

    start_idx = list_of_locations.index(starting_car_location)
    path_to_start = nx.reconstruct_path(next_idx, start_idx, predecessors)
    total_path.extend(path_to_start[1:])
    print(dropoff_mapping)
    return total_path, dropoff_mapping

def visit_clusters(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    G = nx.Graph(incoming_graph_data=adjacency_matrix, cutoff=1000)
    pprint(nx.clustering(G))

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    fixed_adjacency_matrix = np.mat(adjacency_matrix)
    for r in range(fixed_adjacency_matrix.shape[0]):
        for c in range(fixed_adjacency_matrix.shape[1]):

            if fixed_adjacency_matrix[r,c] == 'x':
                #print('x at %d %d'%(r, c))
                if r == c:
                    fixed_adjacency_matrix[r,c] = 0
                else:
                    fixed_adjacency_matrix[r, c] = np.Inf
            else:
                fixed_adjacency_matrix[r,c] = float(fixed_adjacency_matrix[r,c])
    fixed_adjacency_matrix = fixed_adjacency_matrix.astype(np.float)
    return greedyAllPairs(list_of_locations, list_of_homes, starting_car_location, fixed_adjacency_matrix)
    d = {}
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[0])):
            if adjacency_matrix[i][j] > 0 or i==j:
                d[(i,j)] = adjacency_matrix[i][j]
            else:
                d[(i, j)] = float('inf')
    return tsp(list_of_locations, list_of_homes, d)


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)



#############################################################################


def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)


    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs, homeList = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    G = nx.Graph(incoming_graph_data=adjacency_matrix, cutoff=1000)
    predecessors, distances = nx.floyd_warshall_predecessor_and_distance(G)
    print(list(distances.items())[0][1][0])

    prev_dropoff = list_locations.index(starting_car_location)

    list_houses = homeList

    def cost(h, i, j, k):
        '''
        h : place you are coming from
        i : place to drop off person
        j : actual house for person dropped off at i
        k : next place to drive to
        '''
        return ((2 / 3) * list(distances.items())[h][1][i]) + (list(distances.items())[i][1][j]) + (
                    (2 / 3) * list(distances.items())[i][1][k])

    #how do i access the next index in the list of houses?
    #ask eric and calvin to check the code and see if it has correct syntax
    i = 0
    drop_to_home = {}
    dropoff_list = []
    while (i < num_houses - 1):
        i_ind = list_houses.index(list_houses[i])
        i_ind_next = list_houses.index(list_houses[i+1])
        total_cost = list(distances.items())[prev_dropoff][1][i_ind]
        best_dropoff = i_ind
        for x in list_locations:
            x_ind = list_locations.index(x)
            challenge = cost(prev_dropoff, x, i_ind, i_ind_next)
            if challenge < total_cost:
                total_cost = challenge
                best_dropoff = x_ind
        prev_dropoff = best_dropoff
        best_dropoff_location = list_locations[best_dropoff]
        drop_to_home[best_dropoff] = [i_ind]
        dropoff_list.append(best_dropoff_location)
        i +=1
    return greedyAllPairs(list_locations, dropoff_list, drop_to_home, starting_car_location, adjacency_matrix)


    # convertToFile(car_path, drop_offs, output_file, list_locations)



#############################################################################





def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    # mat = [[0, 1, 2, 1],
    #  [1, 0, 1, 1],
    #  [1, 1, 0, 1],
    #  [1, 1, 1, 0]]
    mat = [[0, 10, 8, 1, 0],
           [10, 0, 2, 0, 0],
           [8, 2, 0, 1, 1],
           [1, 0, 1, 0, 4],
           [0, 0, 1, 4, 0]]
    #print(solve(['A', 'B', 'C', 'D', 'E'], ['B', 'E'], 'A', mat))


    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)

    visit_clusters(None, None, None, mat)
    if False:
        #print(solve(['A', 'B', 'C', 'D', 'E'], ['B', 'E'], 'A', mat))


        parser = argparse.ArgumentParser(description='Parsing arguments')
        parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
        parser.add_argument('input', type=str, help='The path to the input file or directory')
        parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
        parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
        args = parser.parse_args()
        output_directory = args.output_directory
        if args.all:
            input_directory = args.input
            solve_all(input_directory, output_directory, params=args.params)
        else:
            input_file = args.input
            solve_from_file(input_file, output_directory, params=args.params)
