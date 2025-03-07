import os
import requests # type: ignore
import urllib.parse
import json
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.neighbors import KDTree # type: ignore
from fastapi import FastAPI, Request, Depends # type: ignore 
from fastapi.responses import JSONResponse, RedirectResponse # type: ignore
from dotenv import load_dotenv # type: ignore
from k_means import k_means_optimal
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Spotify URLs
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"

# Scopes required for fetching user's recently played tracks
SCOPES = "user-read-recently-played"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def login():
    """Redirect user to Spotify authorization page."""
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SCOPES,
        "show_dialog": True  # Forces login every time
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(auth_url)


@app.get("/refresh_token")
def refresh_access_token():
    """Refreshes the access token using refresh_token."""
    refresh_token = TOKEN_STORAGE.get("refresh_token")
    if not refresh_token:
        return JSONResponse({"error": "No refresh token available"}, status_code=400)

    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
    }

    response = requests.post(SPOTIFY_TOKEN_URL, data=token_data)
    token_json = response.json()

    # Update the stored token
    TOKEN_STORAGE["access_token"] = token_json.get("access_token")

    return JSONResponse({"message": "Access token refreshed", "access_token": TOKEN_STORAGE["access_token"]})


TOKEN_STORAGE = {} #for now

# @app.get("/callback")
# def callback(request: Request):
#     """Handles Spotify OAuth callback and exchanges authorization code for access token."""
#     code = request.query_params.get("code")
#     if not code:
#         return JSONResponse({"error": "Authorization failed"}, status_code=400)

#     token_data = {
#         "grant_type": "authorization_code",
#         "code": code,
#         "redirect_uri": SPOTIFY_REDIRECT_URI,
#         "client_id": SPOTIFY_CLIENT_ID,
#         "client_secret": SPOTIFY_CLIENT_SECRET
#     }

#     response = requests.post(SPOTIFY_TOKEN_URL, data=token_data)
#     token_json = response.json()

#     # Store tokens in memory (Replace this with a database for persistence)
#     TOKEN_STORAGE["access_token"] = token_json.get("access_token")
#     TOKEN_STORAGE["refresh_token"] = token_json.get("refresh_token")
#     TOKEN_STORAGE["expires_in"] = token_json.get("expires_in")  # 3600 seconds (1 hour)

#     redirect_url = f"http://localhost:8000/recommendations"
#     return RedirectResponse(url=redirect_url)

@app.get("/callback")
def callback(request: Request):
    """Handles Spotify OAuth callback and exchanges authorization code for access token."""
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "Authorization failed"}, status_code=400)

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
    }

    response = requests.post(SPOTIFY_TOKEN_URL, data=token_data)
    token_json = response.json()

    TOKEN_STORAGE["access_token"] = token_json.get("access_token")
    TOKEN_STORAGE["refresh_token"] = token_json.get("refresh_token")

    return RedirectResponse(url="http://localhost:3000/recommendations")



@app.get("/recently_played")
def get_recently_played():
    """Fetch at least 50 unique songs and their audio features."""
    #the limit is 50, but we can do the last day, or 2 days , based on unix time stamp
    access_token = TOKEN_STORAGE.get("access_token")
    if not access_token:
        return JSONResponse({"error": "User not logged in"}, status_code=401)

    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{SPOTIFY_API_BASE_URL}/me/player/recently-played?limit=50"

    unique_songs = {}  
    next_url = url  

    # Fetch at least 50 unique songs
    while len(unique_songs) < 201 and next_url:
        response = requests.get(next_url, headers=headers)

        if response.status_code == 401:  # Token expired? Refresh it.
            refresh_access_token()
            headers["Authorization"] = f"Bearer {TOKEN_STORAGE['access_token']}"
            response = requests.get(next_url, headers=headers)

        if response.status_code != 200:
            return JSONResponse({"error": "Failed to fetch recently played tracks"}, status_code=response.status_code)

        data = response.json()
        tracks = data.get("items", [])

        for track in tracks:
            song_id = track["track"]["id"]

            if song_id in unique_songs:
                unique_songs[song_id]["count"] += 1
                unique_songs[song_id]["weight"] += 1 * (0.75 ** (unique_songs[song_id]["count"] - 1))
            else:
                unique_songs[song_id] = {
                    "id": song_id,
                    "count": 1,
                    "weight": 1.0
                }

            # if len(unique_songs) >= 200:
            #     break

        next_url = data.get("next")

    return list(unique_songs.values())


@app.get("/refresh_token")
def refresh_token(refresh_token: str):
    """Refreshes the access token using refresh_token."""
    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
    }

    response = requests.post(SPOTIFY_TOKEN_URL, data=token_data)
    token_json = response.json()

    return JSONResponse(token_json)

@app.get("/recommendations")
def get_recommendations():
    recently_listened_df = pd.DataFrame(get_recently_played())
    total_songs_df = pd.read_csv("spotify_data.csv", usecols=["artist_name", "track_name", "track_id", "popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]) #columns are: artist_name, track_name, track_id, popularity, year, genre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, valence, tempo, duration_ms, time_signature
    scaler = StandardScaler() #batch norm for features (also good for PCA because setting mean of each feature to 0)
    numeric_columns_to_normalize = total_songs_df.select_dtypes(include=["number"]).columns
    total_songs_df[numeric_columns_to_normalize] = scaler.fit_transform(total_songs_df[numeric_columns_to_normalize])
    
    total_cols_to_be_projected = total_songs_df.iloc[:, 3:]
    pca = PCA(n_components=3)
    total_cols_to_be_projected_pca = pca.fit_transform(total_cols_to_be_projected)

    total_songs_df[["PC1", "PC2", "PC3"]] = total_cols_to_be_projected_pca

    total_songs_dict =  total_songs_df.set_index("track_id").apply(tuple, axis=1).to_dict()

    recently_listened_df = recently_listened_df[recently_listened_df['id'].isin(total_songs_dict.keys())].reset_index(drop=True)

    feature_names = ["popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "PC1", "PC2", "PC3"] #no need to project anymore i think? or i can project afterwards, unsure
    recently_listened_df[feature_names] = pd.NA

    for i in range(len(recently_listened_df)):
        row = recently_listened_df.iloc[i]
        #if row['id'] in total_songs_dict:
        #print(list(total_songs_dict[row["id"]]))
        features_list = list(total_songs_dict[row["id"]])[2:]
        #print(features_list)
        for x in range(len(features_list)):
            recently_listened_df.loc[i, feature_names[x]] = features_list[x] 

    cluster_centers = k_means_optimal(recently_listened_df[["PC1", "PC2", "PC3"]].to_numpy(), recently_listened_df["weight"].to_numpy())

    tree = KDTree(total_songs_df[["PC1", "PC2", "PC3"]].to_numpy(), metric='euclidean')
    distances, indices = tree.query(cluster_centers, k=5) #take k = 5 here, and check if any of these are in the recently listened_df

    recommendations = []
    for i in range(len(indices)):
        for j in range(len(indices[0])):
            row = total_songs_df.iloc[indices[i][j]]
            recommendations.append({
                "artist": row["artist_name"],
                "track": row["track_name"],
                "url": f"https://open.spotify.com/track/{row['track_id']}"
            })

    return JSONResponse({ "recommendations" : recommendations})    


if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
