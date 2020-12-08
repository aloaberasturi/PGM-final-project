#!/usr/bin/python3
from utils import get_features, get_users, get_edges, get_u_min, get_u_minus
from edge import Edge 
from node import User, Feature, Item
from graph import Graph
import numpy as np

def build_topology(matrix_S, matrix_D, active_user, target_song):

    # 1) *********************** STATIC TOPOLOGY ***********************
    # 1.a) ********************* Content based *************************        
    
    # 1.a.1) Instantiate Acb. 
    a_cb = User(active_user, cb = True)

    # 1.a.2) Instantiate songs reviewed by the active user      
    row_index = matrix_S.loc[matrix_S['user_id'] == active_user].index[0]
    item_nodes = [Item(i) for i in matrix_S.columns[1:] if matrix_S.at[row_index, i] != 0]

    # 1.a.3) Instantiate edges from items rated by active user to A_cb (i --> A_cb)
    i_acb_edges = [Edge(i, a_cb) for i in item_nodes]

    # 1.a.4) Instantiate features and the corresponding edges (f --> i) to their children items
    feature_nodes = get_features(item_nodes, matrix_D)
    f_i_edges = get_edges(item_nodes, matrix_D)

    # 1.b) ********************* Collaborative component *************************       
    # 1.b.1) Instantiate Acf. Instantiate k most-similar users. 
    a_cf = User(active_user, cf = True)
    user_nodes = get_users(active_user, matrix_S, k=4)
 
    # 1.b.2) Instantiate edges from k-most similar users to Acf.
    u_acf_edges = [Edge(u, a_cf) for u in user_nodes]

    # 2) *********************** DYNAMIC TOPOLOGY ***********************    
    # 2.a) ********************* Content based *************************
    # 2.a.1) Set target item 
    # check if it is already instantiated. Otherwise, instantiate
    try:
        target_song_node = [i for i in item_nodes if i.index == target_song][0]
        target_song_node.set_as_target()
    except IndexError:
        target_song_node = Item(target_song, is_target=True)
        item_nodes.append(target_song_node)

    # 2.a.2) Instantiate edges from all features describing target item to target item. 
    target_features_nodes = get_features(target_song_node, matrix_D)
    f_target_song_edges = [Edge(f, target_song_node) for f in target_features_nodes]

    # 2.b) ********************* Collaborative component ************************* 

    # ======= Alejandra
    # 2.b.1) From the set of k-most similar users, get those that didn't rate the target item, U_.     
    # u_minus = get_u_minus([u for u in user_nodes if u.index != active_user], target_song_node, matrix_S)

    # # 2.b.2) Instantiate edges from items rated by users in U_ to users in U_. 
    # i_u_minus_edges = get_edges(u_minus, matrix_S)

    # # Return graph
    # nodes = item_nodes + feature_nodes + user_nodes + [a_cb] + [a_cf] 

    # edges = i_acb_edges + f_i_edges + u_acf_edges + \
    #         f_target_song_edges + i_u_minus_edges

    # graph = Graph(nodes, edges)
    
    # return graph
    
# ======= Federico
    # 2.b.1) From the set of k-most similar users, get those that didn't rate the target item, U_.
    u_min = get_u_min(user_nodes, target_song, item_nodes, matrix_S)
    # 2.b.2) Instantiate edges from items rated by users in U_ to users in U_.
    u_min_edges = get_edges(u_min, matrix_S)

    # Return graph
    nodes = item_nodes + feature_nodes + user_nodes + [a_cb] + [a_cf] 

    edges = i_acb_edges + f_i_edges + u_acf_edges + \
            f_target_song_edges + u_min_edges

    graph = Graph(nodes, edges)
    
    return graph

