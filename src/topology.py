#!/usr/bin/python3
from utils import compute_similarity
from edge import Edge 
from node import Node

def create_topology():


    # 1) *********************** STATIC TOPOLOGY ***********************

    # 1.a) ********************* Content based *************************         
    # 1.a.1) Instantiate Acb. 
    # 1.a.2) Instantiate songs reviewd by active user and their corresponding features
    # 1.a.2) Instantiate edges from all features to children items
    # 1.a.3) Instantiate edges from items rated by active user to Acb. 
    # 1.b) ********************* Collaborative component *************************   
    # 1.b.1) Instantiate Acf. Instantiate k most-similar users. 
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