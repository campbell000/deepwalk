from itertools import product,permutations
from scipy.io import loadmat
from scipy.io import savemat
import scipy
from collections import defaultdict
from six import iteritems
from scipy.sparse import issparse
import numpy as np
import networkx as nx

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}

# Load edgelist and labels as numpy matrices
network = nx.read_edgelist("../example_graphs/email-Eu-core.txt")
numpy_matrix = nx.to_scipy_sparse_matrix(network)
#labels = nx.read_edgelist("../example_graphs/email-Eu-core-department-labels.txt")
with open("../example_graphs/email-Eu-core-department-labels.txt", 'r') as f:
    lines = f.readlines()

a = [[int(a.split()[0]) for a in x.split(" ")] for x in lines]
arr = np.zeros((42, 1005))
for e in a:
    arr[e[1]][e[0]] = 1.0

arr = arr.transpose()


#a = np.transpose(a)
matdict = {"network":numpy_matrix, "group":scipy.sparse.csc_matrix(arr)}

# Save as mat file
savemat("../example_graphs/email.mat", matdict)

mat = loadmat("../example_graphs/email.mat")
A = mat["network"]
graph = sparse2graph(A)
labels_matrix = mat["group"]
print(labels_matrix.A)
labels_count = labels_matrix.shape[1]
print("ASD")