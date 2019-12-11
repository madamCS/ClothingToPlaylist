# IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

# FOR THE SPOTIFY API 
import spotipy
import requests
import base64
import spotipy.util as util
import json 
from spotipy.oauth2 import SpotifyClientCredentials

# TODO: REPLACE WITH YOUR SPOTIFY DEVELOPER CREDENTIALS
client_id = ""
client_secret = ""

# GET AN ACCESS TOKEN WITH YOUR CREDENTIALS
# (NEEDED FOR ALL API REQUESTS)
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
access_token = token.get_access_token()
s = spotipy.Spotify(access_token)

# INITIALIZE AUTHORIZATION HEADERS
headers = {
        "Authorization": "Bearer {}".format(access_token),
        'Content-Type': 'application/json'
    }

# PLAYLISTS RELATED TO CLOTHING
playlist_autumnacoustic = "37i9dQZF1DWUNIrSzKgQbP"
playlist_sunnyday = "37i9dQZF1DX1BzILRveYHb"
playlist_darkandstormy = "37i9dQZF1DX2pSTOxoPbx9?si=bnOf9I1lRR-Bg4o53MRMCw"
playlist_sadvibe = "37i9dQZF1DXaJZdVx8Fwkq"
playlist_workday = "37i9dQZF1DXcsT4WKI8W8r"
playlist_ragebeats = "37i9dQZF1DX3ND264N08pv"

# LIST OF PLAYLIST IDs IN ORDER
playlist_ids = [playlist_autumnacoustic,
               playlist_sunnyday,
               playlist_darkandstormy,
               playlist_sadvibe,
               playlist_workday,
               playlist_ragebeats]

# LIST OF PLAYLIST NAMES IN ORDER
playlist_names = ["AUTUMN ACOUSTIC",
                 "SUNNY DAY",
                 "DARK AND STORMY",
                 "SAD VIBE",
                 "WORK DAY",
                 "RAGE BEATS"]


# INITIALIZE WARMTH INDEX & TEMPERATURE FOR EACH PLAYLIST
playlists_warmth_index = [1, 4, 2, 3, 2.5, 4]

# AVERAGE TEMPERATURE FOR PLAYLIST SCENARIOS
playlists_temperatures = [53.9, 69.53, 56, 39, 45, 66.7]

# TURN TEMPERATURE INTO A 1-5 INDEX
# (Y POINT FOR OUR DATA)
def get_temp_index(temp):
    return temp / 20.0

# INDEX PLAYLIST TEMPS:
indexed_playlist_temps = [get_temp_index(x) for x in playlists_temperatures]

# HELPER FUNCTIONS FOR USER INPUTS

# LIST OF ALL THE USER'S CLOTHING
# INPUT: dictionary of items user is wearing
def get_articles_of_clothing(clothing):
    articles_of_clothing = []
    for key, val in clothing.items():
        if val:
            articles_of_clothing.append(key)
        
    return articles_of_clothing

# GETS WARMTH INDEX BASED ON WHAT USER IS WEARING
# INPUT: list of the articles of clothing user is wearing
# INPUT: dictionary of warmth indexed items
def get_index_of_clothing(articles_of_clothing, warmth_index):
    index = 0
    for item in articles_of_clothing:
        index += warmth_index[item]
        
    return index / len(articles_of_clothing)

## USER INPUTS !!

# TODO: REPLACE WITH URL TO ONE OF YOUR PLAYLISTS
# DEMO PLAYLIST:
my_playlist = ""

# TODO: REPLACE WITH YOUR USER SPOTIFY ID
USER_ID = ""

# TODO: REPLACE WITH T/F VALUES BASED ON YOUR OUTFIT
# DICTIONARY OF CLOTHING VALUES
clothing = {
    "JACKET" : True,
    "BRIGHT_COLORS" : False,
    "SANDALS": False,
    "BOOTS" : True,
    "SCARF" : False,
    "BLAZER": False,
    "DARK_COLORS": True
}

# TODO: REPLACE WITH THE CURRENT TEMPERATURE OUTSIDE (IN ºF)
TEMPERATURE_OUTSIDE = None

# WARMTH INDEX OF CLOTHING - SET IN STONE
# (can alter if desired)
warmth_index = {
    "JACKET" : 0,
    "BRIGHT_COLORS" : 4,
    "SANDALS" : 5,
    "BOOTS" : 2,
    "SCARF" : 1,
    "BLAZER" : 3,
    "DARK_COLORS" : 3
}

# GET ARTICLES OF CLOTHING & INDEX FOR USER
articles = get_articles_of_clothing(clothing)
index = get_index_of_clothing(articles, warmth_index)
index_temperature = get_temp_index(TEMPERATURE_OUTSIDE)

# REGRESSION MODEL: SET TRAINING & TESTING DATA
# SET X_TRAIN DATA
data = {}
for i in range(len(playlist_names)):
    data.update({playlist_names[i]: [playlists_warmth_index[i], indexed_playlist_temps[i]]})
    
pd_data = pd.DataFrame(data=data)
X_train = pd_data.loc[0] # X = clothing index
y_train = pd_data.loc[1] # Y = indexed temperature

# user-given clothing index
x_test = index
# user-given indexed temperature
y_test = index_temperature

# reshape data
X_train = np.array(X_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

errors = []
predictions = []
for k in range(1, len(playlist_names)):
    model = neighbors.KNeighborsRegressor(n_neighbors = k)
    model.fit(X_train, y_train) # fit on X_train, y_train
    y_hat = model.predict(x_test)
    predictions.append(y_hat)
    # get error (RMSE)
    error = sqrt(mean_squared_error([y_test], y_hat))
    errors.append(error)

# PICK BEST K
min_error = min(errors)
best_k = errors.index(min_error) + 1

# GET PREDICTION OF BEST K
best_y_hat = predictions[best_k - 1]

# NOW, GET THE PLAYLIST !!
# predicted best_y_hat = temperature index prediction
# so, which playlist is this closest to?
distance_from_playlist = [abs(y_hat[0] - x) for x in playlists_warmth_index]
shortest_distance = min(distance_from_playlist)
shortest_distance_index = distance_from_playlist.index(shortest_distance)

best_playlist = playlist_names[shortest_distance_index]
best_playlist_id = playlist_ids[shortest_distance_index]


# HELPER FUNCTIONS FOR SPOTIFY API

# RETURNS A PLAYLIST'S JSON RESPONSE GIVEN ITS ID
def get_playlist_tracks(playlist_id):
    endpoint = "https://api.spotify.com/v1/playlists/%s/tracks" % playlist_id
    res = requests.get(endpoint, headers = headers)
    return res.json()

# PARSE THE PLAYLIST DATA
# GET ALL OF THE ARTIST & ALBUM IDs
def get_playlist_ids(playlist_response):
    album_ids = []
    artist_ids = []
    artists_name = []
    for i in playlist_response["items"]:
        #print("ITEM IS:", i, "\n")
        # MAKE THE UNICODE OBJECT A STRING
        for id_val in i["track"]["artists"][0]["id"].split("\n"):
            artist_ids.append(str(id_val))
        for name in i["track"]["artists"][0]["name"].split("\n"):
            artists_name.append(name)
        album_ids.append(str(i['track']['album']['id']))
        
    return artist_ids, artists_name, album_ids

## GET A PLAYLIST'S ID FROM LINK
## (we will need this to transform our user's input)
def playlist_id_from_link(playlist_link):
    # This is what a link looks like: 
    # https://open.spotify.com/user/spotify/playlist/37i9dQZF1DX3ND264N08pv?si=Ict0LQLATdmwRIhv8ew0mg
    junk, playlist, temp1 = playlist_link.partition("/playlist/")
    playlist_id, q, junk = temp1.partition("?")
    return playlist_id

# GET RELATED ARTISTS
# INPUT: a list of artist IDs
def get_related_artists(artist_ids):
    related_artists_ids = []
    related_artists_names = []

    for artist in artist_ids:
        endpoint = "https://api.spotify.com/v1/artists/%s/related-artists" % artist
        res = requests.get(endpoint, headers=headers).json()
        if res["artists"]:
            for a in res["artists"][0]["id"].split("\n"):
                related_artists_ids.append(a)
            for n in res["artists"][0]["name"].split("\n"):
                related_artists_names.append(n)
                
    return related_artists_ids, related_artists_names

## FUNCTION WHICH FINDS SIMILAR ARTISTS PER. 2 PLAYLISTS
# artists_ids_user : artist IDs from user's playlist
# new_playlist_id : playlist id we want to cross-check with
# num_times_recurse : counter on the #times recursion happens
# recurstion_limit : limit on the amount of recursion
def cross_check_artists(artist_ids_user, artists_ids_best_playlist, num_times_recurse, recursion_limit = 3):
    # SET A LIMIT ON THE AMOUNT OF RECURSION 
    if num_times_recurse == recursion_limit:
        print("TOO MUCH RECURSION, REACHED LIMIT. NO RELATED ARTISTS.")
        return 0, recursion_limit
    
    # CROSS CHECK THE NAMES OF ARTISTS
    similar_names = set(artist_ids_user) & set(artists_ids_best_playlist)
    
    # IF NO SIMILAR NAMES, RECURSE
    if len(similar_names) == 0:
        # GO ONE LAYER DEEPER, RELATED ARTISTS OF USER'S RELATED ARTISTS
        num_times_recurse += 1
        new_artists_ids, new_artists_names = get_related_artists(artists_ids_user)
        cross_check_artists(new_artists_ids, artists_ids_best_playlist, num_times_recurse)
        return
    
    # ELSE
    return similar_names, num_times_recurse

# GET ARTIST'S TOP TRACKS
def get_artist_top_tracks(artist_id):
    endpoint = "https://api.spotify.com/v1/artists/%s/top-tracks?country=US" % artist_id
    res = requests.get(endpoint, headers=headers).json()
    
    num_top_tracks = len(res["tracks"])
    track_ids = []
    for i in range(num_top_tracks):
        track_ids.append(str(res["tracks"][i]["id"]))
        
    return track_ids

# GET SEED ARTISTS 
# (5 RANDOM ARTIST IDs FROM LIST)
# we are going to use the related artists w highest frequency
def get_seed_artists(related_artists, related_artists_frequency):
    seed_artists = []
    while len(seed_artists) < 5:
        max_frequency = max(related_artists_frequency)
        max_frequency_index = related_artists_frequency.index(max_frequency)
        max_frequency_artist = related_artists[max_frequency_index]
        
        if str(max_frequency_artist) in seed_artists:
            continue
            
        seed_artists.append(str(max_frequency_artist))
        
        # ADJUST BEFORE CALLING RECURSION
        related_artists_frequency.pop(max_frequency_index)
        related_artists.pop(max_frequency_index)
        
    return seed_artists # LIST OF ARTIST IDs
    
# GET SEED TRACKS 
# (5 TRACK IDs FOR RECOMMENDATION)
# we are going to use five random tracks from best_playlist
def get_seed_tracks(target_playlist_id):
    seed_tracks = []
    res_target = get_playlist_tracks(target_playlist_id)
    
    while len(seed_tracks) < 5:
        items = res_target["items"]
        len_items = len(items) # 50 items

        # PICK A RANDOM ITEM
        random_item = random.choice(items)
        track_id = str(random_item["track"]["id"])
        
        if track_id in seed_tracks:
            continue
            
        seed_tracks.append(track_id)
    
    return seed_tracks

# NOW, WE CAN GENERATE THE PLAYLIST

# INITIALIZE SONGS IN PLAYLIST
SONGS_IN_PLAYLIST = []

# GET USER'S PLAYLIST INFORMATION
user_res = get_playlist_tracks(user_playlist_id)
user_artists_ids, user_artists_names, user_album_ids = get_playlist_ids(user_res)
user_related_artists_ids, user_related_artists_names = get_related_artists(user_artists_ids)

# GET BEST PLAYLIST'S INFORMATION
best_playlist_res = get_playlist_tracks(best_playlist_id)
best_playlist_artists_ids, best_playlist_artists_names, best_playlist_album_ids = get_playlist_ids(best_playlist_res)
best_playlist_related_artists_ids, best_playlist_related_artists_names = get_related_artists(best_playlist_artists_ids)

# FIND SIMILAR ARTISTS BETWEEN THE TWO PLAYLISTS
similar_artists = cross_check_artists(user_artists_ids, best_playlist_artists_ids, 0)
if len(similar_artists) < 1:
    print("NO SIMILAR ARTISTS FOUND :(")

# IF SIMILAR ARTISTS EXIST, PULL SONGS FROM THEM
endpoint = "https://api.spotify.com/v1/playlists/%s/tracks" % best_playlist_id
tracks = requests.get(endpoint, headers=headers).json()
num_tracks = len(tracks["items"])
        
if len(similar_artists) >= 1:
    for i, a in enumerate(similar_artists[0]):
        # A = ARTIST ID
        # FIND THEIR SONGS IN BEST_PLAYLIST
        for t in range(num_tracks):
            track_artist_id = tracks["items"][t]["track"]["artists"][0]["id"]
            if track_artist_id == a:
                # IF THIS IS AN ARTIST'S SONG, ADD THE TRACK
                SONGS_IN_PLAYLIST.append(str(tracks["items"][t]["track"]["id"]))
                
                if len(SONGS_IN_PLAYLIST) >= 10:
                    break

# IF PLAYLIST IS NOT FILLED (10 SONGS):
if len(similar_artists) >= 1:
    while len(SONGS_IN_PLAYLIST) < 10:
    # TRY: top tracks of similar_artists
        for i, a in enumerate(similar_artists[0]):
            top_tracks = get_artist_top_tracks(a)
            # PICK AT RANDOM
            random_track = random.choice(top_tracks)
            SONGS_IN_PLAYLIST.append(random_track)

# IF NO RELATED ARTISTS
# PICK 5 RANDOM SONGS FROM EACH PLAYLIST
# DIG ONE LAYER DEEPER FOR EACH ( SO ALWAYS SOMETHING NEW ) based on FREQUENCY of related artists
# FREQUENCY OF RELATED ARTISTS FOR USER PLAYLIST
related_artists_ids, related_artists_names = get_related_artists(user_artists_ids)
related_artists_frequency = [related_artists_names.count(name) for name in related_artists_names]

# FREQUENCY OF RELATED ARTISTS FOR BEST PLAYLIST
best_related_artists_ids, best_related_artists_names = get_related_artists(best_playlist_artists_ids)
best_related_artists_frequency = [best_related_artists_names.count(name) for name in best_related_artists_names]

# THEN, 5 TRACKS BASED ON FREQUENCY OF RELATED ARTISTS
def get_5_tracks(frequency, related_artists_ids):
    count = 0
    while count < 5:
        most_freq = max(frequency)
        most_freq_artist_index = frequency.index(most_freq) 
        most_freq_artist_id = related_artists_ids[most_freq_artist_index]
        most_freq_artist_id_index = related_artists_ids.index(most_freq_artist_id)
        print(most_freq_artist_id)
        
        # GET THAT ARTIST'S TOP TRACKS & PICK RANDOM
        track_ids = get_artist_top_tracks(most_freq_artist_id)
        track = random.choice(track_ids)
        SONGS_IN_PLAYLIST.append(track)
        
        # REMOVE USED ITEMS
        frequency.remove(most_freq)
        related_artists_ids.remove(most_freq_artist_id)
        
        count += 1

# GET 5 TRACKS FROM EACH PLAYLIST'S DATA
if len(SONGS_IN_PLAYLIST) == 0:
    get_5_tracks(related_artists_frequency, related_artists_ids)      
    get_5_tracks(best_related_artists_frequency, best_related_artists_ids)

# CREATE A NEW PLAYLIST IN USER'S ACCOUNT
"""
WHERE TO GET THE PLAYLIST AUTH TOKEN:
https://developer.spotify.com/console/post-playlists/?body=%7B%22name%22%3A%22New%20Playlist%22%2C%22description%22%3A%22New%20playlist%20description%22%2C%22public%22%3Afalse%7D
"""
playlist_auth_token = ""
endpoint_url = "https://api.spotify.com/v1/users/%s/playlists" % USER_ID

create_playlist_headers = {
        "Authorization": "Bearer {}".format(playlist_auth_token),
        'Content-Type': 'application/json'
    }

request_body = json.dumps({
          "name": "CLOTHING TO PLAYLIST",
          "description": "Songs you should listen to based on what you're wearing.",
          "public": True
        })

created_playlist = requests.post(url=endpoint_url, data=request_body, headers=create_playlist_headers)
created_playlist_id = str(created_playlist.json()['id'])

# PREPARE TRACK IDS TO BE ADDED TO PLAYLIST
# TRACKS MUST BE INI URI FORMAT: spotify:track:id
def get_track_uris(list_of_track_ids):
    URIS = []
    for i in range(len(list_of_track_ids)):
        URIS.append("spotify:track:" + list_of_track_ids[i])
    return URIS

URIS = get_track_uris(SONGS_IN_PLAYLIST)

# ADD TRACKS TO YOUR PLAYLIST
def add_tracks_to_playlist(playlist_id):
    endpoint_url = "https://api.spotify.com/v1/playlists/%s/tracks" % playlist_id

    add_tracks_body = json.dumps({
        "uris": URIS
    })
    
    add_tracks_res = requests.post(url=endpoint_url, data=add_tracks_body, headers=create_playlist_headers)
    res = add_tracks_res.json()
    return res
    
add_tracks_res = add_tracks_to_playlist(created_playlist_id)
snapshot_id = str(add_tracks_res['snapshot_id'])

# GENERATE PLAYLIST LINK
playlist_link = "https://open.spotify.com/user/%s/playlist/%s" % (USER_ID, created_playlist_id)
print("START LISTENING:", playlist_link)
