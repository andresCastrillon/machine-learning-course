#%% md
# # K-Means Clustering
#%% md
# ## Importing the libraries
#%% md
# 
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% md
# ## Importing the dataset
#%%
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
#%% md
# ## Using the elbow method to find the optimal number of clusters
#%%
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title( 'the Elbow Method' )
plt.xlabel( 'Number of clusters' )
plt.ylabel('WCSS')
plt.show()
#%% md
# ## Training the K-Means model on the dataset
#%%
model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_means = model.fit_predict(X)

print(y_means)
#%% md
# ## Visualising the clusters
#%%
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label='Cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', label='Cluster 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label='Cluster 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'yellow', label='Cluster 4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'brown', label='Cluster 5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'black', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of customers')
plt.show()
#%%
