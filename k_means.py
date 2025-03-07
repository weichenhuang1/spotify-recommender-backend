from sklearn.cluster import KMeans #this can handle wighted datasets
from sklearn.metrics import silhouette_score

#WITHOUT WEIGHTED SILHOUETTE SCORE
def k_means_optimal(X, weights):
    num_rows = len(X[0])
    last_silhouette_score = None #we keep going until the silhouette scores stop increasing by at least 3%
    for x in range(2, num_rows + 1):
        kmeans = KMeans(n_clusters=x, init = 'k-means++', random_state=21)
        kmeans.fit(X, sample_weight=weights)
        curr_silhouette_score = silhouette_score(X, kmeans.labels_) #we can try without weighted silhouette score for now
        #print(f"cluster_centers {kmeans.cluster_centers_}")
        #print(f"silhouette score {curr_silhouette_score}")
        if last_silhouette_score:
            if ((curr_silhouette_score - last_silhouette_score) / last_silhouette_score ) < 0.2:
                #we have hit the stopping criteria
                #return [x, kmeans.labels_, kmeans.cluster_centers_]
                return kmeans.cluster_centers_
        if x == num_rows:
            #return [x, kmeans.labels_, kmeans.cluster_centers_]
            return kmeans.cluster_centers_

    return #a list of cluster centers list[list[]] #inner list is the dimensions

#TESTING HERE
# import pandas as pd
# recently_listened_df = pd.read_csv("D:\\school\\projects\\spotify\\recs\\recently_listened_df.csv")

# cluster_centers = k_means_optimal(recently_listened_df[["PC1", "PC2", "PC3"]].to_numpy(), recently_listened_df["weight"].to_numpy())


# print(cluster_centers)
# print(type(cluster_centers))


# #ok might as well try the other stuff here too
# from sklearn.neighbors import KDTree
# import numpy as np
# import pandas as pd

# total_songs_df = pd.read_csv("D:\\school\\projects\\spotify\\recs\\total_songs_with_pca.csv")
# np.random.seed(21)
# print(total_songs_df[["PC1", "PC2", "PC3"]].to_numpy())
# tree = KDTree(total_songs_df[["PC1", "PC2", "PC3"]].to_numpy(), metric='euclidean')
# distances, indices = tree.query(cluster_centers, k=5) #take k = 5 here, and check if any of these are in the recently listened_df

# print(indices)
# print(type(indices))
# for i in range(len(indices)):
#     for j in range(len(indices[0])):
#         row = total_songs_df.iloc[indices[i][j]]
#         print(f"{row['artist_name']} {row['track_name']} open.spotify.com/track/{row['track_id']}")