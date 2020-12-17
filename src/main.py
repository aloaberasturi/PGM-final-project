#!/usr/bin/python3

from topology import build_topology
from pathlib import Path
from inference import perform_inference
import topologyUtils as utils
import pandas as pd

if __name__ == '__main__':
    """
    * Matrix_D has songs as rows and features as columns.
    * Matrix_S has users as rows and songs as columns
    """
    # A) read songs with features, users and ratings from db
    [music_data, user_data, rating_data] = utils.get_data()
    [songs_dict, users_dict, features_dict] = utils.get_dicts(music_data, user_data, rating_data)

    # B) create matrix S and D    
    matrix_D = music_data.drop('name', axis=1).rename(columns={v:k for (k,v) in features_dict.items()}) 
    matrix_S = utils.compute_scores(user_data, music_data, rating_data) 
    
    # C) choose active user and target song
    [active_user, target_song] = utils.select_user_and_song(matrix_S)  

    # D) build topology
    graph = build_topology(matrix_S, matrix_D, active_user, target_song)

    # E) compute ratings    
    rating = perform_inference(graph, matrix_D, matrix_S)

    # F) classify rating
    user_opinion = utils.check_rating(rating)

    print("User %s would rate song %s with a %i, with likelihood of %d percent." % (active_user, target_song, rating[0],rating[1]))
    print("The user might think that %s" % user_opinion)

    active_user_name = list(users_dict.keys())[list(users_dict.values()).index(active_user)]
    target_song_name = list(songs_dict.keys())[list(songs_dict.values()).index(target_song)]

    print("What %s might think about %s: %s" % (active_user_name,target_song_name,user_opinion))
