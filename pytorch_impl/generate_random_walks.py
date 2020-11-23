import torch
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

import numpy as np
np.random.seed(123)
import networkx as nx
import node2vec
import pandas as pd

def read_graph(input, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G
def read_KG(input):
    data=pd.read_csv(input,sep='\t',names=['s','r','t'])
    # data=data.iloc[:1000]
    G=nx.from_pandas_edgelist(data, "s", "t", edge_attr=None, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G=G.to_undirected()
    return G
def generate_random_walks(input, num_walks, walk_length,kg=False):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    
    if kg:
        nx_G=read_KG(input)
    else:
        nx_G = read_graph(input)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)  #DeepWalk
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    return np.array(walks)
