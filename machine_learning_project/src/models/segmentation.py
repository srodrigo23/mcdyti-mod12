# segmentation.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_segmentation(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_data)

    return data

def evaluate_segmentation(data: pd.DataFrame) -> float:
    if 'cluster' not in data.columns:
        raise ValueError("DataFrame must contain 'cluster' column for evaluation.")
    
    silhouette_avg = silhouette_score(data.drop('cluster', axis=1), data['cluster'])
    return silhouette_avg

def get_segmented_data(data: pd.DataFrame, cluster_label: int) -> pd.DataFrame:
    if 'cluster' not in data.columns:
        raise ValueError("DataFrame must contain 'cluster' column to filter by cluster.")
    
    return data[data['cluster'] == cluster_label]