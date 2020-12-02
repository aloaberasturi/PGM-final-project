#!/usr/bin/python3
import pandas as pd
from pathlib import Path

def get_data():
    data_folder = Path('../' + "data")
    music_data = data_folder / "music_data.csv"
    music_data = pd.read_csv(music_data)
    users_data = pd.read_csv('../data/users_data.csv')
    ratings_data = pd.read_csv('../data/ratings.csv')
    music_data['original_title'] = music_data['original_title'].str.lower()
    music_data= music_data.drop(columns=['song_id', 'blabla','year'])
    return [music_data, users_data, ratings_data]

def compute_similarity(u1,u2):
    pass

def compute_weights(x,y):
    pass
    # check class of x and y 
    # compute w(f,i), w(i,u) or w(u,a)

def compute_description_matrix(data):
    pass

def compute_score_matrix(data):
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