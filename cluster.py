import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# 1. Przetwarzanie danych
# Read countries
df = pd.read_csv("countries of the world.csv", sep = "," , encoding= 'utf-8')

# Select the relevant columns for clustering (chyba wszstkie oprócz nazwy i miejsca?)
columns = ['Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)', 'Net migration',
           'Infant mortality (per 1000 births)', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
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

# 2. k means clustering
k = 3  # dane w zadaniu
kmeans = KMeans(n_clusters=k, random_state=123)
kmeans.fit(scaled_data)

# Assign cluster labels to the data
cluster_labels = kmeans.labels_

# Add cluster labels to the original data
df['Cluster'] = cluster_labels

