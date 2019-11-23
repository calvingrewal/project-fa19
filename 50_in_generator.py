import networkx as nx
import numpy as np
import scipy as sp

G = nx.Graph()

for i in range (50):
    G.add_node(i)

j = 0
while (j < 38):
    for k in range (j+1, j+6):
        G.add_edge(j, k, weight=1)
    G.add_edge(j, j+6, weight=2)
    for l in range (j+6, j+14):
        G.add_edge(l, l+1, weight=2)
    G.add_edge(j+13, j+14, weight=3)
    j += 14
for m in range(42, 49):
    G.add_edge(m, m+1, weight=1)
G.add_edge(49, 0, weight=4)
mat = sp.sparse.csr_matrix.toarray(nx.adjacency_matrix(G))
print(mat)
np.savetxt("input2.in", mat, fmt="%d")
