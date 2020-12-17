#!/usr/bin/python3
from node import User, Item, Feature
from edge import Edge
import topologyUtils 

class Graph():
    """
    A class for graph objects. Stores and retrieves everythin necessary to work with the graph
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        # types of nodes
        self.user_nodes = [u for u in nodes if isinstance(u, User)]
        self.item_nodes = [i for i in nodes if isinstance(i, Item)]
        self.feature_nodes = [f for f in nodes if isinstance(f, Feature)]

        # types of edges
        self.item_user_edges = [i_u for i_u in edges if (isinstance(i_u.x, Item) and isinstance(i_u.y, User))]
        self.feature_item_edges = [f_i for f_i in edges if (isinstance(f_i.x, Feature) and isinstance(f_i.y, Item))]
        self.u_acf_edges = [u_acf for u_acf in edges if (isinstance(u_acf.x, User) and isinstance(u_acf.y, User))]

    # *********** Getters ***********
    def get_user_nodes(self):
        return self.user_nodes

    def get_item_nodes(self):
        return self.item_nodes
    
    def get_feature_nodes(self):
        return self.feature_nodes

    def get_a_cf(self):       
        return next((u for u in self.user_nodes if u.is_cf), None)
    
    def get_a_cb(self):
        return next((u for u in self.user_nodes if u.is_cb), None)   
    
    def get_target_item(self):        
        return next ((i for i in self.item_nodes if i.is_target), None)

    def get_target_features(self):
        return [edge.x for edge in self.feature_item_edges if edge.y.is_target]   

    def get_parents(self, node):
        if (isinstance(node, User) and not node.is_cf):
            return [e.x for e in self.item_user_edges if e.y == node] 
        elif (isinstance(node, User) and node.is_cf):
            return [e.x for e in self.u_acf_edges if e.y == node]
        else:
            return [e.x for e in self.feature_item_edges if e.y == node]

    def get_children(self, node): 
        if (isinstance(node, User) and not node.is_cf):
            return [e.y for e in self.item_user_edges if e.x == node]
        elif (isinstance(node, User) and node.is_cf):
            return [e.y for e in self.u_acf_edges if e.x == node]
        elif (isinstance(node, Feature)):
            return [e.y for e in self.feature_item_edges if e.x == node]
        elif (isinstance(node, Item)):
            return [e.y for e in self.item_user_edges if e.x == node]
