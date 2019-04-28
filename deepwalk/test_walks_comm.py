# Tests correctness of random walks

import networkx as nx
import matplotlib.pyplot as plt
import networkx
from graph import *
from networkx.generators.community import *

G = nx.connected_caveman_graph(5, 5)
deepwalk_G = from_networkx_forreal(G)
random_walk = deepwalk_G.mbrw_random_walk(15, start=1, bias_val=100)
edges = G.edges()
for edge in edges:
    found = False
    for i in range(len(random_walk)):
        if (i < len(random_walk) - 1):
            if (random_walk[i] == str(edge[0]) and random_walk[i+1] == str(edge[1])) or \
                    (random_walk[i] == str(edge[1]) and random_walk[i+1] == str(edge[0])):
                found = True

    if found:
        G[edge[0]][edge[1]]['color'] = 'red'
        G[edge[0]][edge[1]]['weight'] = 4
    else:
        G[edge[0]][edge[1]]['color'] = 'grey'
        G[edge[0]][edge[1]]['weight'] = 1

color_map = ['yellow' for x in G.nodes]

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
edges,colors = zip(*nx.get_edge_attributes(G,'color').items())
plt.figure(figsize=(18,18))
plt.title("Memory-Biased Random Walk of Path: "+str(random_walk))
nx.draw(G, node_color=color_map, node_size=500, edgelist=edges, with_labels=True, edge_color=colors, width=weights)
plt.show()






