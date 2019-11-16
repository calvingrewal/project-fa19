import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import numpy as np, pandas as pd
from more_itertools import iterate, take
from pulp import LpProblem, LpVariable, LpBinary, lpDot, lpSum, value

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""


def tsp(nodes, dist=None):
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
    a = pd.DataFrame([(i, j, dist[i, j])
                      for i in range(n) for j in range(n) if i != j], columns=['NodeI', 'NodeJ', 'Dist'])

    m = LpProblem()

    # creates x_ij for every edge in the graph
    a['VarIJ'] = [LpVariable('x%d' % i, cat=LpBinary) for i in a.index]

    a['VarJI'] = a.sort_values(['NodeJ', 'NodeI']).VarIJ.values

    #
    u = [0] + [LpVariable('y%d' % i, lowBound=0) for i in range(n - 1)]

    # gives the total distance,
    m += lpDot(a.Dist, a.VarIJ)

    for _, v in a.groupby('NodeI'):
        m += lpSum(v.VarIJ) == 1  # constraint for one edge exiting each vertex
        m += lpSum(v.VarJI) == 1  # constraint for one edge entering each vertex

    for _, (i, j, _, vij, vji) in a.query('NodeI!=0 & NodeJ!=0').iterrows():
        m += u[i] + 1 - (n - 1) * (1 - vij) + (n - 3) * vji <= u[j]  # 持ち上げポテンシャル制約(MTZ)
    for _, (_, j, _, v0j, vj0) in a.query('NodeI==0').iterrows():
        m += 1 + (1 - v0j) + (n - 3) * vj0 <= u[j]  # 持ち上げ下界制約
    for _, (i, _, _, vi0, v0i) in a.query('NodeJ==0').iterrows():
        m += u[i] <= (n - 1) - (1 - vi0) - (n - 3) * v0i  # 持ち上げ上界制約
    m.solve()
    a['ValIJ'] = a.VarIJ.apply(value)
    dc = dict(a[a.ValIJ > 0.5][['NodeI', 'NodeJ']].values)
    return value(m.objective), list(take(n, iterate(lambda k: dc[k], 0)))

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
        A list of (location, [homes]) representing drop-offs
    """
    return tsp(adjacency_matrix_to_edge_list(adjacency_matrix))


    

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

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)
    
    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    mat = [[0, 1, 1, 1],
     [1, 0, 1, 1],
     [1, 1, 0, 1],
     [1, 1, 1, 0]]

    print(solve(None, None, None, mat))

    if False:
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


        
