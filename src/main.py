#!/usr/bin/python3

from topology import create_topology
# from inference import perform_inference
import utils
import pandas as pd

if __name__ == '__main__':

    # A) read songs with features, users and ratings from db

    [music_data, user_data, rating_data] = utils.get_data()
    [songs_dict, users_dict, features_dict] = utils.get_dicts(music_data, user_data, rating_data)

    # B) create matrix S and matrix D    

    matrix_D = music_data.drop('name', axis=1).rename(columns={v:k for (k,v) in features_dict.items()})
    
    import numpy as np
    song_ids = [song_id for song_id in songs_dict.values()]  # maybe there is a more efficient way to get these instead of
    user_ids = [user_id for user_id in users_dict.values()]  # taking them out of the dictionary?
    matrix_S = pd.DataFrame(np.zeros((len(user_ids), len(song_ids))),index=user_ids, columns=song_ids)  # create zero matrix (set r = 0 for all values)
    #print(matrix_S)   # showing 0 matrix
    for ind,row in rating_data.iterrows():       # for each index, row in the rating data
        matrix_S[row['song_id']][row['user_id']] = row['rating'] # set each value in matrix_S to corresponding rating if exists, otherwise stays 0
    #print(matrix_S)  # matrix was updated
    
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


