import os
import requests # type: ignore
import urllib.parse
import pandas as pd # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.neighbors import KDTree # type: ignore
from fastapi import FastAPI, Request, Depends # type: ignore 
from fastapi.responses import JSONResponse, RedirectResponse # type: ignore
from dotenv import load_dotenv # type: ignore
from k_means import k_means_optimal
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import plotly.colors as pc #type: ignore
import time

#so to make the viz: i need recommendations_df, recently_listened_df, and centroids_df
app = FastAPI()

load_dotenv()
TOKEN_STORAGE = {} #for now
CACHE_STORAGE = {
    "recently_listened": None,  # Cached DataFrame
    "recommendations": None,  # Cached Recommendations
    "recommendations_df" : None,
    "cluster_results" : None,  #cluster labels, cluster centers
    "timestamps": {}  # Timestamp to track cache expiry
}

CACHE_EXPIRY = 3600

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://spotify-recommender-frontend-pink.vercel.app"],  # Allows React frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_total_songs():
    total_songs_df = pd.read_csv("spotify_data.csv", usecols=[
        "artist_name", "track_name", "track_id", "popularity",
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence"
    ])
    
    scaler = StandardScaler()
    numeric_columns_to_normalize = total_songs_df.select_dtypes(include=["number"]).columns
    total_songs_df[numeric_columns_to_normalize] = scaler.fit_transform(total_songs_df[numeric_columns_to_normalize])

    pca = PCA(n_components=3)
    total_songs_df[["PC1", "PC2", "PC3"]] = pca.fit_transform(total_songs_df.iloc[:, 3:])

    return total_songs_df

TOTAL_SONGS_DF = load_total_songs()  

def is_cache_expired(cache_key):
    """Check if the cache for a given key is expired."""
    if cache_key not in CACHE_STORAGE["timestamps"]:
        return True
    return (time.time() - CACHE_STORAGE["timestamps"][cache_key]) > CACHE_EXPIRY



@app.get("/recently_played")
def get_recently_played():
    """Fetch at least 50 unique songs and their audio features."""
    #the limit is 50, but we can do the last day, or 2 days , based on unix time stamp
    access_token = TOKEN_STORAGE.get("access_token")
    if not access_token:
        return JSONResponse({"error": "User not logged in"}, status_code=401)

    if CACHE_STORAGE["recently_listened"] is not None and not is_cache_expired("recently_listened"):
        return CACHE_STORAGE["recently_listened"]

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
            track_name = track["track"]["name"]
            artist_name = track["track"]["artists"][0]["name"]

            if song_id in unique_songs:
                unique_songs[song_id]["count"] += 1
                unique_songs[song_id]["weight"] += 1 * (0.75 ** (unique_songs[song_id]["count"] - 1))

            else:
                unique_songs[song_id] = {
                    "id": song_id,
                    "count": 1,
                    "weight": 1.0,
                    "track_name" : track_name,
                    "artist_name" : artist_name
                }

            # if len(unique_songs) >= 200:
            #     break

        next_url = data.get("next")

    CACHE_STORAGE["recently_listened"] = list(unique_songs.values())
    CACHE_STORAGE["timestamps"]["recently_listened"] = time.time()
    
    return CACHE_STORAGE["recently_listened"]


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

    return RedirectResponse(url="https://spotify-recommender-frontend-pink.vercel.app/recommendations")



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
    """Fetch and cache song recommendations."""
    if CACHE_STORAGE["recommendations"] is not None and not is_cache_expired("recommendations"):
        return CACHE_STORAGE["recommendations"]

    recently_listened_df = pd.DataFrame(get_recently_played())
    
    # Create a dictionary for faster lookup
    total_songs_dict = TOTAL_SONGS_DF.set_index("track_id").apply(tuple, axis=1).to_dict()
    recently_listened_df = recently_listened_df[recently_listened_df['id'].isin(total_songs_dict.keys())].reset_index(drop=True)
    CACHE_STORAGE["recently_listened"] = recently_listened_df
    feature_names = ["popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "PC1", "PC2", "PC3"]
    recently_listened_df[feature_names] = pd.NA

    for i in range(len(recently_listened_df)):
        row = recently_listened_df.iloc[i]
        features_list = list(total_songs_dict[row["id"]])[2:]
        for x in range(len(features_list)):
            recently_listened_df.loc[i, feature_names[x]] = features_list[x] 
    if CACHE_STORAGE["cluster_results"] is not None and not is_cache_expired("cluster_results"):
        cluster_centers = CACHE_STORAGE["cluster_results"][1]
    else:
        CACHE_STORAGE["cluster_results"] =  k_means_optimal(recently_listened_df[["PC1", "PC2", "PC3"]].to_numpy(), recently_listened_df["weight"].to_numpy())
        cluster_centers = CACHE_STORAGE["cluster_results"][1]

    tree = KDTree(TOTAL_SONGS_DF[["PC1", "PC2", "PC3"]].to_numpy(), metric='euclidean')
    distances, indices = tree.query(cluster_centers, k=5)
    recommendations_df = pd.DataFrame(columns = ["track_name", "artist_name", "PC1", "PC2", "PC3", "cluster"])

        
    recommendations = []
    for i in range(len(indices)):
        for j in range(len(indices[0])):
            row = TOTAL_SONGS_DF.iloc[indices[i][j]]
            recommendations_df.loc[len(recommendations_df)] = [row["track_name"], row["artist_name"], row["PC1"], row["PC2"], row["PC3"], i]
            recommendations.append({
                "artist": row["artist_name"],
                "track": row["track_name"],
                "url": f"https://open.spotify.com/track/{row['track_id']}"
            })

    CACHE_STORAGE["recommendations"] = {"recommendations": recommendations}
    CACHE_STORAGE["timestamps"]["recommendations"] = time.time()
    CACHE_STORAGE["recommendations_df"] = recommendations_df

    return CACHE_STORAGE["recommendations"]

#rip from gpt, we shall see
#so k_means_optimal  returns a list of [x, cluster_labels, cluster centroids]
#and i want to plot each point in the recently_listened_df on the plot, with the song name visible, as well as color for the cluster it belongs
#I also want the centroid for each cluster
@app.get("/plot_clusters")
def plot_clusters():
    recently_listened_df = pd.DataFrame(CACHE_STORAGE["recently_listened"])
    cluster_labels = CACHE_STORAGE["cluster_results"][0]
    print(cluster_labels)
    print(type(cluster_labels[0]))
    print(recently_listened_df)

    cluster_centers = CACHE_STORAGE["cluster_results"][1]

    recently_listened_df["cluster"] = cluster_labels
    centroids_df = pd.DataFrame(cluster_centers, columns=["PC1", "PC2", "PC3"])
    centroids_df["cluster"] = centroids_df.index

    recommendations_df = CACHE_STORAGE["recommendations_df"]


    fig = go.Figure()
    sorted_clusters = sorted(recently_listened_df['cluster'].unique())  # Ensure correct ordering
    cluster_colors = pc.qualitative.Set1[:len(sorted_clusters)]  
    cluster_color_map = {cluster: color for cluster, color in zip(sorted_clusters, cluster_colors)}

    for cluster in recently_listened_df['cluster'].unique():
        cluster_df = recently_listened_df[recently_listened_df['cluster'] == cluster]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_df['PC1'], y=cluster_df['PC2'], z=cluster_df['PC3'],
            mode='markers',
            marker=dict(size=6, symbol='circle', opacity=0.7, color=cluster_color_map[cluster]),
            name=f'Recently Listened - Cluster {cluster}',
            customdata=cluster_df[['track_name', 'artist_name']],  
            hoverinfo="none",
            hovertemplate="<b>%{customdata[0]}</b><br>by %{customdata[1]}<extra></extra>"
        ))



    # # ========== Add Cluster Centroids (Color-Coded) ========== #
    for cluster in centroids_df['cluster'].unique():
    #     cluster_df = centroids_df[centroids_df['cluster'] == cluster]
        centroid_df = centroids_df[centroids_df['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=centroid_df['PC1'], y=centroid_df['PC2'], z=centroid_df['PC3'],
            mode='markers',
            marker=dict(size=10, symbol='x', color=cluster_color_map[cluster]),
            name=f'Centroid - Cluster {cluster}',
            hoverinfo="none"
        ))


    # # ========== Add Recommended Songs ========== #
    for cluster in recommendations_df['cluster'].unique():
        rec_df = recommendations_df[recommendations_df['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=rec_df['PC1'], y=rec_df['PC2'], z=rec_df['PC3'],
            mode='markers',
            marker=dict(
                size=6, 
                symbol='diamond', 
                color=cluster_color_map[cluster],  # Same color as cluster
                line=dict(color='black', width=2)  # Thin black outline
            ),
            name=f'Recommended - Cluster {cluster}',
            customdata=rec_df[['track_name', 'artist_name']],
            hoverinfo="none",
            hovertemplate="<b>%{customdata[0]}</b><br>by %{customdata[1]}<extra></extra>"
        ))

    # ========== Enable Click Mode ========== #
    fig.update_layout(
        title="3D Scatter of Songs, Centroids, and Recommendations",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        clickmode='event+select',  # Enable click interaction
        legend=dict(
            title="Legend",
            x=0.8,  # Move legend to the right
            y=1.0
        )
    )

    # Show figure
    return JSONResponse(content=fig.to_json())  # Convert to JSON and return


if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)