#!/usr/bin/python3

from topology import build_topology
from pathlib import Path

from inference import perform_inference
import utils
import pandas as pd

if __name__ == '__main__':

    # A) read songs with features, users and ratings from db
    [music_data, user_data, rating_data] = utils.get_data()
    [songs_dict, users_dict, features_dict] = utils.get_dicts(music_data, user_data, rating_data)

    # B) create matrix S and D    
    # matrix_D has songs as rows and features as columns
    matrix_D = music_data.drop('name', axis=1).rename(columns={v:k for (k,v) in features_dict.items()})
    # matrix_S has users as rows and songs as columns
    matrix_S = utils.compute_scores(user_data, music_data, rating_data)    
    
    # C) choose active user and target song
    [active_user, target_song] = utils.select_user_and_song(matrix_S)  

    # D) build topology
    graph = build_topology(matrix_S, matrix_D, active_user, target_song)

    # E) compute ratings    
    rating = perform_inference(graph, matrix_D, matrix_S, item_instantiation=False) # still not tested if True

    # F) classify rating
    # user_opinion = utils.check_rating(rating)

    # print("User %s would rate song %s with a %i" % (active_user, target_song, rating))
    # print("The user might think that %s" % user_opinion)


