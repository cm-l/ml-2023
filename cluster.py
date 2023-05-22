import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 1. Przetwarzanie danych
# Read countries
df = pd.read_csv("countries of the world.csv", sep=",", encoding='utf-8')

# Select the relevant columns for clustering (chyba wszstkie oprócz nazwy i miejsca?)
columns = ['Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
           'Net migration',
           'Infant mortality (per 1000 births)', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)',
           'Arable (%)',
           'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

data = df[columns]

# Regex do , --> . w string na float
data = data.replace(',', '.', regex=True)

# Convert data to float
data = data.astype(float)

# Puste
data = data.dropna()

# Reset index to match the dropped rows
data.reset_index(drop=True, inplace=True)

# Reset index of the original DataFrame
df = df.loc[data.index].reset_index(drop=True)

# Mierzymy odległości więc standaryzujemy je
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 1.5 Szukanie dobrego k
# Silhouette Coefficient
max_score = -1
best_k = 2  # Minimum k (bez 1)
for k in range(2, 25):  # Range of k
    kmeans = KMeans(n_clusters=k, random_state=555555)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    score = silhouette_score(scaled_data, labels)
    if score > max_score:  # max(a,b)
        max_score = score
        best_k = k
print("Znalezione sugerowane k:" + str(best_k))

# 2. k means clustering
k = 3  # dane w zadaniu
kmeans = KMeans(n_clusters=k, random_state=123)
kmeans.fit(scaled_data)

# Assign cluster labels to the data
cluster_labels = kmeans.labels_

# Add cluster labels to the original data
df['Cluster'] = cluster_labels

# Podsumowanie wykonanej analizy
for cluster in range(k):
    cluster_data = df[df['Cluster'] == cluster]
    cluster_size = len(cluster_data)
    cluster_center = kmeans.cluster_centers_[cluster]
    print(f"-- Cluster {cluster + 1} --\n")
    print(f"Ile krajów?: {cluster_size}")
    print(f"Centroid: {cluster_center}")
    print("5 pierwszych krajów:")
    print(cluster_data.head())
    print("\n----------------------------\n")

# 3. Dendrogram
linkage_matrix = linkage(scaled_data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode="level", labels=cluster_labels, color_threshold=k)
plt.title('Plot Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster')
plt.ylabel('Odległość (euklidesowa)')
plt.tight_layout()

# zawartosc
plt.show()

# Plot the dendrogram 2.0 redux
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode="level", labels=cluster_labels, color_threshold=3, p=k)
plt.title('Plot Hierarchical Clustering Dendrogram (wersja 1:1 z dokumentacji)')
plt.xlabel('Number of points in node (or index if no ())')
plt.ylabel('Odległość (euklidesowa)')
plt.tight_layout()

# zawartosc
plt.show()