#!/usr/bin/python3

from topology import create_topology
from pathlib import Path
# from inference import perform_inference
import utils
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # A) read songs with features, users and ratings from db

    [music_data, user_data, rating_data] = utils.get_data()
    [songs_dict, users_dict, features_dict] = utils.get_dicts(music_data, user_data, rating_data)

    # B) create matrix S and D    

    matrix_D = music_data.drop('name', axis=1).rename(columns={v:k for (k,v) in features_dict.items()})
    user_ids = user_data.user_id.values.tolist()
    song_ids = music_data.song_id.values.tolist()
    matrix_S = pd.DataFrame(np.zeros((len(user_ids), len(song_ids))), index=user_ids, columns=song_ids)
    for _, row in rating_data.iterrows():
        # set each value in matrix_S to corresponding rating if exists, otherwise stays 0
        matrix_S[row['song_id']][row['user_id']] = row['rating'] 
    
    # C) choose active user and target song

    [active_user, target_song] = utils.select_user_and_song(matrix_S)  

    # D) create topology

    graph = create_topology(matrix_S, matrix_D, active_user, target_song)

    # E) compute ratings
    
    rating = perform_inference(matrix_S, matrix_D, graph)

    # F) classify rating

    user_opinion = utils.check_rating(rating)

    print("User %s would rate song %s with a %i" % (active_user, target_song, rating))
    print("The user might think that %s" % user_opinion)


