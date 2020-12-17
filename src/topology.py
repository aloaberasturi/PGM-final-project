#!/usr/bin/python3
from topologyUtils import get_users, get_edges, get_u_min, get_u_minus, get_u_plus
from edge import Edge 
from node import Node, User, Feature, Item
from graph import Graph
import numpy as np
import gc

def build_topology(matrix_S, matrix_D, active_user, target_song):
    """
    Function that builds the topology of the network.
    Parameters
    ----------

    matrix_S: pd.DataFrame
    matrix_D: pd.DataFrame
    active_user: int
                Active user ID
    target_song: int
                Target song ID
    """

    # 1) *********************** STATIC TOPOLOGY ***********************
    # 1.a) ********************* Content based *************************        
    
    # 1.a.1) Instantiate Acb. 
    a_cb = User(active_user, cb = True)

    # 1.a.2) Instantiate songs reviewed by the active user      
    item_nodes = [Item(i) for i in matrix_S.columns[1:]]

    # 1.a.3) Instantiate edges from items rated by active user to A_cb (i --> A_cb)
    i_acb_edges = [Edge(i, a_cb) for i in item_nodes]

    # 1.a.4) Instantiate features and the corresponding edges (f --> i) to their children items
    feature_nodes = [Feature(f) for f in matrix_D.columns[1:]]
    f_i_edges = get_edges(feature_nodes, item_nodes, matrix_D)

    # 1.b) ********************* Collaborative component *************************       
    # 1.b.1) Instantiate Acf. Instantiate k most-similar users. 
    a_cf = User(active_user, cf = True)
    user_nodes = get_users(active_user, matrix_S, k=4)
 
    # 1.b.2) Instantiate edges from k-most similar users to Acf.
    u_acf_edges = [Edge(u, a_cf) for u in user_nodes]

    # 2) *********************** DYNAMIC TOPOLOGY ***********************    
    # 2.a) ********************* Content based *************************
    # 2.a.1) Set target item 
    target_song_node = [i for i in item_nodes if i.index == target_song][0]
    target_song_node.set_as_target()

    # 2.a.2) Instantiate edges from all features describing target item to target item. 
    f_target_song_edges = get_edges(feature_nodes, [target_song_node], matrix_D)

    # 2.b) ********************* Collaborative component ************************* 

    # 2.b.1) From the set of k-most similar users, get those that didn't rate the target item, U_.     
    u_minus = get_u_minus([u for u in user_nodes if u.index != active_user], target_song_node, matrix_S)

    # 2.b.2) Instantiate edges from items rated by users in U_ to users in U_. 
    i_u_minus_edges = get_edges(item_nodes, u_minus, matrix_S)

    # 3) Return graph
    nodes = [obj for obj in gc.get_objects() if isinstance(obj, Node)]

    edges = [obj for obj in gc.get_objects() if isinstance(obj, Edge)]

    graph = Graph(nodes, edges)
    
    return graph

