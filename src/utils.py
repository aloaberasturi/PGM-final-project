#!/usr/bin/python3
import pandas as pd
import random
from pathlib import Path
import numpy as np

def get_data():

    data_folder = Path('../' + "data")

    music_data = data_folder / "music_data.csv"
    music_data = pd.read_csv(music_data)
    music_data.dropna(inplace = True) 
    music_data['name'] = music_data['name'].str.lower()
    user_data = data_folder / "user_data.csv"
    user_data = pd.read_csv(user_data)
    user_data['name'] = user_data['name'].str.lower()
    user_data.dropna(inplace = True) 
    rating_data = data_folder / "ratings.csv"    
    rating_data = pd.read_csv(rating_data)
    rating_data.dropna(inplace = True) 

    return [music_data, user_data, rating_data]

def get_dicts(music_data, user_data, rating_data):

    songs_dict = pd.Series(music_data.song_id.values, index = music_data.name).to_dict() #maybe here?
    users_dict = pd.Series(user_data.user_id.values, index = user_data.name).to_dict()
    features = list(music_data.keys()[2:])
    keys = ["f_%i" % i for i in range(len(features))]
    features_dict = {k:v for (k, v) in zip(keys, features)}

    return [songs_dict, users_dict, features_dict]

def compute_scores(user_data, music_data, rating_data):
    user_ids = user_data.user_id.values.tolist()
    song_ids = music_data.song_id.values.tolist()
    matrix_S = pd.DataFrame(np.zeros((len(user_ids), len(song_ids))), index=user_ids, columns=song_ids)

    for _, row in rating_data.iterrows():
        # Set each value in matrix_S to corresponding rating if exists, otherwise stays 0
        matrix_S[row['song_id']][row['user_id']] = row['rating']

    return matrix_S

def compute_similarity(u1,u2):
    pass

def get_k_nn(matrix_S, active_user):
    pass

def compute_weights(matrix_D, matrix_S, active_user):
    # parameters m and n_k used in w(f,i)
    m = len(matrix_D['song_id'])
    frequencies = {'n_{}'.format(i): sum(matrix_D['f_%s' % i]) for i in range(len(matrix_D.keys()[1:]))}

    # parameter iucb == |I(Ucb)| used in w(i,u)    
    row = matrix_S[matrix_S['user_id'] == active_user]
    iucb = row.drop(['user_id'], axis=1).values.sum(axis=1)

def select_user_and_song(matrix_S):
    while True:
        user_index = random.randint(0, len(matrix_S.axes[0]) - 1)
        song_index = random.randint(0, len(matrix_S.axes[1]) - 1)
        rating = matrix_S.iloc[user_index, song_index]
        if (rating ==0):
            break
    song = matrix_S.columns.values[song_index]
    user = matrix_S.index.values[user_index]
    return (user, song)


def check_rating(rating):
    if (rating < 3):
        opinion = 'the song is awful! :('
    elif (3 <= rating < 5): 
        opinion = 'the song is pretty bad :/'
    elif (5 <= rating < 7): 
        opinion = 'the song is fine :)'
    elif (7 <= rating < 9): 
        opinion = "the song is very interesting ^^"
    elif (9 <= rating < 10):
        opinion = 'the song is brilliant :D '
    elif (rating == 10):
        opinion = 'the song is memorable <3'
    
    return opinion