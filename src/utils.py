#!/usr/bin/python3
import pandas as pd
from pathlib import Path

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

    songs_dict = pd.Series(music_data.song_id.values, index = music_data.name).to_dict()
    users_dict = pd.Series(user_data.user_id.values, index = user_data.name).to_dict()
    features = list(music_data.keys()[2:])
    keys = ["f_%i" % i for i in range(len(features))]
    features_dict = {k:v for (k, v) in zip(keys, features)}

    return [songs_dict, users_dict, features_dict]


def compute_similarity(u1,u2):
    pass

def compute_weights(x,y):
    pass
    # check class of x and y 
    # compute w(f,i), w(i,u) or w(u,a)

def compute_description_matrix(music_data):
    pass

def compute_score_matrix(user_data, rating_data):
    pass

def input_query(user,song, matrix_S):
    pass
    # user has already rated song according to matrix_S, return error
    # select active user and target item 

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