#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.items()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]


  # Avoids visiting the same node twice
  def self_avoiding_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      # Generate list of neighbors to visit, but remove already-visited nodes in candidates
      cur = path[-1]
      nodes_to_visit = G[cur]
      for visited_node in path:
        nodes_to_visit = remove_values_from_list(nodes_to_visit, visited_node)

      if len(nodes_to_visit) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(nodes_to_visit))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

  # Disallows the node from backtracking (visiting previous node)
  def no_backtracking_random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      nodes_to_visit = G[cur]

      # If list's size is >= 2, remove the second-to-last node from the list of candidate nodes to visit
      if len(nodes_to_visit) >= 2:
        nodes_to_visit = remove_values_from_list(nodes_to_visit, nodes_to_visit[-1])

      if len(nodes_to_visit) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(nodes_to_visit))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

  # Memory biased: Fair random walks are based on having NO information. This models societies more accurately.
  # From http://simlab.biomed.drexel.edu/papers_published/yucel_networks.pdf
  def mbrw_random_walk(self, path_length, memory_size=5, bias_val=1000, alpha=0, rand=random.Random(), start=None):
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      nodes_to_visit = G[cur]

      # Get all neighbors, but remove the node we just visited
      if len(nodes_to_visit) >= 2:
        nodes_to_visit = remove_values_from_list(nodes_to_visit, nodes_to_visit[-2])

      # If no valid neighbors exist, stop. Otherwise, proceed with the algorithm
      if len(nodes_to_visit) == 0:
        break
      else:
        # Get the <memory_size> most recent nodes that we've visited (get last 5, but exclude the most recent since that's
        # where we're currently at)
        memory_buffer = path[-memory_size-1:-1]

        # init default probabilities to visit each node
        probabilities = [(1/(bias_val + len(nodes_to_visit))) for x in nodes_to_visit]

        # check to see if a candidate neighbor is in our memory buffer. Take the LAST ocurrence, since this will be the most
        # recent node visited
        neighbor_index_in_memory = -1
        for node_in_memory in enumerate(memory_buffer):
          if node_in_memory in nodes_to_visit:
            neighbor_index_in_memory = nodes_to_visit.index(nodes_to_visit)

        # if one of the neighbors is in memory, make that probability MUCH larger
        if neighbor_index_in_memory != -1:
          probabilities[neighbor_index_in_memory] = bias_val / (bias_val + len(nodes_to_visit))
        # Otherwise, normalize the probabilities so they add up to 1, and are all the same
        else:
          prob_total = sum(probabilities)
          probabilities = [(x / probabilities) for x in probabilities]

        # Finally, choose the neighbor to go to based on the probabilities we constructed
        path.append(np.random.choice(nodes_to_visit, p=probabilities))

    return [str(node) for node in path]


# TODO add build_walks in here
def do_random_walk(walk_selection, G, path_length, rand, alpha, node):
  if walk_selection == "uniformly_random":
    return G.random_walk(path_length, rand=rand, alpha=alpha, start=node)
  elif walk_selection == "self_avoiding":
    return G.self_avoiding_random_walk(path_length, rand=rand, alpha=alpha, start=node)
  elif walk_selection == "no_backtracking":
    return G.no_backtracking_random_walk(path_length, rand=rand, alpha=alpha, start=node)
  elif walk_selection == "mbrw":
    return G.no_backtracking_random_walk(path_length, rand=rand, alpha=alpha, start=node)
  else:
    raise Exception("INVALID WALK SELECTION: "+str(walk_selection))

def build_deepwalk_corpus(walk_selection, G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(do_random_walk(walk_selection, G, path_length, rand, alpha, node))

  return walks


def build_deepwalk_corpus_iter(walk_selection, G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield do_random_walk(walk_selection, G, path_length, rand, alpha, node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()
  
  total = 0 
  with open(file_) as f:
    for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
      adjlist.extend(adj_chunk)
      total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)

def from_networkx_forreal(G_input, undirected=True):
  G = Graph()

  for idx, x in enumerate(G_input.nodes()):
    for y in iterkeys(G_input[x]):
      G[x].append(y)

  if undirected:
    G.make_undirected()

  return G

def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

def remove_values_from_list(the_list, val):
  return [value for value in the_list if value != val]