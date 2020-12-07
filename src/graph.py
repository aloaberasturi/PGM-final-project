#!/usr/bin/python3
from node import User, Item, Feature
from edge import Edge
import utils 

class Graph():
    def __init__(self, nodes, edges):
        # types of nodes
        self.user_nodes = [u for u in nodes if isinstance(u, User)]
        self.item_nodes = [i for i in nodes if isinstance(i, Item)]
        self.feature_nodes = [f for f in nodes if isinstance(f, Feature)]

        # types of edges
        self.item_user_edges = [i_u for i_u in edges if (isinstance(i_u.x, Item) and isinstance(i_u.y, User))]
        self.feature_item_edges = [f_i for f_i in edges if (isinstance(f_i.x, Feature) and isinstance(f_i.y, Item))]
        self.u_acf_edges = [u_acf for u_acf in edges if (isinstance(u_acf.x, User) and isinstance(u_acf.y, User))]

    def get_active_user(self):       
        return next((u for u in self.user_nodes if u.is_cf == True), None)
    
    def get_u_plus(self, matrix_S):
        target_item = self.get_target_item()
        active_user_index = self.get_active_user().index
        u_minus_indexes = [u_.index for u_ in utils.get_u_minus(self.user_nodes, target_item, matrix_S)]
        u_plus = [u for u in self.user_nodes if (u.index not in u_minus_indexes and u.index != active_user_index)]
        for u in u_plus:
            u.set_rating(matrix_S, self.get_target_item())
        return u_plus

    def get_target_item(self):        
        return next ((i for i in self.item_nodes if i.is_target), None)
    
    def get_target_features(self):
        return [edge.x for edge in self.feature_item_edges if edge.y.is_target]
