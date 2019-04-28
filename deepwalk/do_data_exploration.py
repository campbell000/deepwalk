import igraph
import numpy as np
import networkx as nx
import scipy.io
from igraph import *
import matplotlib.pyplot as plt
import networkx

def load_mat_as_networkx(filepath):
    aa=scipy.io.loadmat(filepath)
    mat=aa['network']
    return nx.from_scipy_sparse_matrix(mat)

def get_stats(g, name):
    print("\n\nGetting Statistics for: "+name)
    print("Number of Nodes: "+str(networkx.number_of_nodes(g)))
    print("Number of Edges: "+str(networkx.number_of_edges(g)))
    print("Avg Clustering Coefficient: "+str(networkx.average_clustering(g)))

g = networkx.read_edgelist("../example_graphs/email-Eu-core.txt")
get_stats(g, "Email")
g = load_mat_as_networkx("../example_graphs/flickr.mat")
get_stats(g, "Flickr")
g = load_mat_as_networkx("../example_graphs/blogcatalog.mat")
get_stats(g, "BlogCatalog")
