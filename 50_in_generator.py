import networkx as nx
import numpy as np
import scipy as sp

G = nx.Graph()

for i in range (50):
    G.add_node(i)

j = 0
while (j < 38):
    for k in range (j, j+5):
        G.add_edge(j, k, weight=1)
    for l in range (j+5, j+12):
        G.add_edge(l, l+1, weight=2)
    G.add_edge(j+11, j+12, weight = 3)
    j += 12

G.add_edge(j, 0, weight=4)
mat = sp.sparse.csr_matrix.toarray(nx.adjacency_matrix(G))
print(mat)
np.savetxt("input2.in", mat, fmt="%d")
