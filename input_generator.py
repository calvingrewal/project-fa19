import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import networkx as nx
import numpy as np, pandas as pd
from more_itertools import iterate, take
from pulp import LpProblem, LpVariable, LpBinary, lpDot, lpSum, value
import scipy

from student_utils import *
def diamond_generator(num_vertices, num_homes):
	M = 100000
	D = nx.Graph()
	for i in range(1, num_vertices):
		D.add_node(i)
	for i in range(1, num_vertices):
		if i % 8 == 1:
			for j in range(i, i + 7):
				D.add_edge(j, j + 1, weight=2)
			D.add_edge(i, i + 4, weight = 1)
			D.add_edge(i + 3, i + 7, weight = 1)
		if i % 8 == 0:
			D.add_edge(i, i + 1, weight = 2)
	for i in range(1, num_vertices):
		if i % 8 == 3:
			for j in range (6, i, 8):
				if i != j:
					D.add_edge(i, j, weight = (int) (nx.dijkstra_path_length(D, i, j)/2 - 1))
			for j in range (i + 11, num_vertices,8):
				if (i == 11):
						print(nx.dijkstra_path_length(D, i, j))
				D.add_edge(i, j, weight = (int) (nx.dijkstra_path_length(D, i, j)/2 - 1))
	D.add_edge(1, num_vertices, weight = 1)
	matrix = nx.adjacency_matrix(D)
	f = open("input1.in", "w")
	length = len(nx.nodes(D))
	f.write(str(length) + "\n")
	f.write(str(length/2) + "\n")
	node_list = nx.nodes(D)
	for i in node_list:
		f.write(str(i) + " ")
	f.write("\n")
	for i in node_list:
		curr = i % 8
		if curr == 2 or curr == 4 or curr == 5 or curr == 7:
			f.write(str(i) + " ")
	f.write("\n")
	f.write("1")
	array = scipy.sparse.csr_matrix.toarray(matrix)
	print(array)
	np.savetxt("matrix", array, fmt = "%d")
	f.close()
diamond_generator(200, 100)

