import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

def compute_distance_matrix(embeddings, metric='cosine'):
    """
    Compute pairwise distance matrix for embeddings.
    
    Parameters:
    - embeddings: array of shape (n_samples, n_features)
    - metric: 'cosine', 'euclidean', 'manhattan', or 'correlation'
    
    Returns:
    - distance_matrix: symmetric matrix of pairwise distances
    """
    if metric == 'cosine':
        # Convert cosine similarity to distance
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
    elif metric == 'euclidean':
        distance_matrix = euclidean_distances(embeddings)
    elif metric == 'correlation':
        # Correlation distance (1 - correlation coefficient)
        distances = pdist(embeddings, metric='correlation')
        distance_matrix = squareform(distances)
    elif metric == 'manhattan':
        distances = pdist(embeddings, metric='manhattan') 
        distance_matrix = squareform(distances)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return distance_matrix

def find_most_similar_timepoints(embeddings, dates_pd, reference_idx, metric='cosine', n_similar=5):
    """
    Find the most similar timepoints to a reference timepoint.
    
    Parameters:
    - embeddings: embedding vectors
    - dates_pd: pandas datetime series
    - reference_idx: index of reference timepoint
    - metric: distance metric to use
    - n_similar: number of most similar points to return
    
    Returns:
    - similar_indices: indices of most similar timepoints
    - distances: corresponding distances
    - dates: corresponding dates
    """
    # Compute distances from reference point to all others
    reference_embedding = embeddings[reference_idx].reshape(1, -1)
    
    if metric == 'cosine':
        similarities = cosine_similarity(reference_embedding, embeddings)[0]
        distances = 1 - similarities
    elif metric == 'euclidean':
        distances = euclidean_distances(reference_embedding, embeddings)[0]
    else:
        distances = cdist(reference_embedding, embeddings, metric=metric)[0]
    
    # Sort by distance (excluding self)
    distance_indices = np.argsort(distances)
    similar_indices = distance_indices[distance_indices != reference_idx][:n_similar]
    
    return similar_indices, distances[similar_indices], dates_pd[similar_indices]

def analyze_pre_post_fire_similarity(embeddings, dates_pd, fire_date, metric='cosine'):
    """
    Analyze similarity between pre-fire and post-fire embeddings.
    
    Returns:
    - pre_fire_centroid: mean embedding of pre-fire period  
    - post_fire_centroid: mean embedding of post-fire period
    - centroid_distance: distance between centroids
    - within_pre_distances: average pairwise distance within pre-fire period
    - within_post_distances: average pairwise distance within post-fire period
    - cross_distances: average distance between pre-fire and post-fire points
    """
    pre_fire_mask = dates_pd < fire_date
    post_fire_mask = dates_pd >= fire_date
    
    pre_fire_embeddings = embeddings[pre_fire_mask]
    post_fire_embeddings = embeddings[post_fire_mask]
    
    # Compute centroids
    pre_fire_centroid = np.mean(pre_fire_embeddings, axis=0)
    post_fire_centroid = np.mean(post_fire_embeddings, axis=0)
    
    # Distance between centroids
    if metric == 'cosine':
        centroid_distance = 1 - cosine_similarity([pre_fire_centroid], [post_fire_centroid])[0,0]
    elif metric == 'euclidean':
        centroid_distance = euclidean_distances([pre_fire_centroid], [post_fire_centroid])[0,0]
    else:
        centroid_distance = cdist([pre_fire_centroid], [post_fire_centroid], metric=metric)[0,0]
    
    # Within-group distances
    if len(pre_fire_embeddings) > 1:
        pre_fire_dist_matrix = compute_distance_matrix(pre_fire_embeddings, metric)
        within_pre_distances = np.mean(pre_fire_dist_matrix[np.triu_indices_from(pre_fire_dist_matrix, k=1)])
    else:
        within_pre_distances = 0
        
    if len(post_fire_embeddings) > 1:
        post_fire_dist_matrix = compute_distance_matrix(post_fire_embeddings, metric)
        within_post_distances = np.mean(post_fire_dist_matrix[np.triu_indices_from(post_fire_dist_matrix, k=1)])
    else:
        within_post_distances = 0
    
    # Cross-group distances
    if len(pre_fire_embeddings) > 0 and len(post_fire_embeddings) > 0:
        if metric == 'cosine':
            cross_similarities = cosine_similarity(pre_fire_embeddings, post_fire_embeddings)
            cross_distances = np.mean(1 - cross_similarities)
        elif metric == 'euclidean':
            cross_distances = np.mean(euclidean_distances(pre_fire_embeddings, post_fire_embeddings))
        else:
            cross_distances = np.mean(cdist(pre_fire_embeddings, post_fire_embeddings, metric=metric))
    else:
        cross_distances = np.nan
    
    return {
        'pre_fire_centroid': pre_fire_centroid,
        'post_fire_centroid': post_fire_centroid, 
        'centroid_distance': centroid_distance,
        'within_pre_distances': within_pre_distances,
        'within_post_distances': within_post_distances,
        'cross_distances': cross_distances
    }

def find_anomalous_timepoints(embeddings, dates_pd, metric='cosine', threshold_percentile=95):
    """
    Find timepoints that are anomalous (most dissimilar to the rest).
    
    Parameters:
    - embeddings: embedding vectors
    - dates_pd: pandas datetime series  
    - metric: distance metric
    - threshold_percentile: percentile threshold for anomaly detection
    
    Returns:
    - anomalous_indices: indices of anomalous timepoints
    - anomaly_scores: anomaly scores (mean distance to all other points)
    """
    n_points = len(embeddings)
    anomaly_scores = np.zeros(n_points)
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(embeddings, metric)
    
    # For each point, compute mean distance to all other points
    for i in range(n_points):
        # Exclude self-distance (which is 0)
        other_distances = np.concatenate([distance_matrix[i, :i], distance_matrix[i, i+1:]])
        anomaly_scores[i] = np.mean(other_distances)
    
    # Find anomalous points based on threshold
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    anomalous_indices = np.where(anomaly_scores >= threshold)[0]
    
    return anomalous_indices, anomaly_scores

def print_similarity_analysis(embeddings, dates_pd, fire_date, metric='cosine'):
    """Print comprehensive similarity analysis results"""
    
    print(f"=== Similarity Analysis using {metric.upper()} distance ===\n")
    
    # Pre/post fire analysis
    results = analyze_pre_post_fire_similarity(embeddings, dates_pd, fire_date, metric)
    
    print(f"Centroid distance (pre vs post-fire): {results['centroid_distance']:.4f}")
    print(f"Average within pre-fire distances: {results['within_pre_distances']:.4f}")
    print(f"Average within post-fire distances: {results['within_post_distances']:.4f}")
    print(f"Average cross-group distances: {results['cross_distances']:.4f}")
    
    # Anomaly detection
    anomalous_indices, anomaly_scores = find_anomalous_timepoints(embeddings, dates_pd, metric)
    
    print(f"\nTop 5 most anomalous timepoints:")
    top_anomalous = np.argsort(anomaly_scores)[-5:][::-1]
    for idx in top_anomalous:
        print(f"  {dates_pd[idx].strftime('%Y-%m-%d')}: score = {anomaly_scores[idx]:.4f}")
    
    return results, anomaly_scores