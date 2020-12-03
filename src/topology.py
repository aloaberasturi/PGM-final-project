#!/usr/bin/python3
from utils import get_k_nn
from edge import Edge 
from node import User, Feature, Item

def build_topology(matrix_S, matrix_D, active_user, target_song):


    # 1) *********************** STATIC TOPOLOGY ***********************

    # 1.a) ********************* Content based *************************   
      
    # 1.a.1) Instantiate Acb. 
    a_cb = User(active_user, cb = True)

    # 1.a.2) Instantiate songs reviewed by the active user... 
    item_nodes = [ Item(i) for i in matrix_S.loc[[active_user]].columns.tolist() if matrix_S.loc[active_user, i]!=0 ]

    # ... and their corresponding features!
    for i in item_nodes:
        item_id = i.index
        a = matrix_D.loc[matrix_D['song_id'] == item_id].columns.tolist()

    # feature_ids = 



    # 1.a.2) Instantiate edges from al features to children items
    # 1.a.3) Instantiate edges from items rated by active user to Acb. 
    # 1.b) ********************* Collaborative component *************************   
    # 1.b.1) Instantiate Acf. Instantiate k most-similar users. 

    a_cf = User(active_user, cf = True)
    k_nn = get_k_nn(matrix_S, active_user)
    user_nodes = [User(user_id) for user_id in k_nn]

    # 1.b.2) Instantiate edges from k-most similar users to Acf.

    # 2) *********************** DYNAMIC TOPOLOGY ***********************
    
    # 2.a) ********************* Content based *************************
    # 2.a.1) Select target item and instantiate it. 
    # 2.a.2) Instantiate edges from all features describing target item to target item. 
    # 2.b) ********************* Collaborative component ************************* 
    # 2.b.1) From the set of k-most similar users, get those that didn't rate the target item, U_. 
    # 2.b.2) Instantiate edges from items rated by users in U_ to users in U_. 

    graph = [nodes, edges]
    return graph