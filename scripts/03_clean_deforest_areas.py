import pandas as pd
import rootutils

root_path = rootutils.find_root()

import pandas as pd
import numpy as np
from shapely import wkt
from scipy.spatial.distance import cdist
import geopandas as gpd

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth in km
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def filter_by_distance(df_tile, min_distance_km=5):
    """
    Keep points that are at least min_distance_km apart, preserving original indices.
    Prioritizes larger areas (size_pixels).
    """
    if len(df_tile) == 0:
        return df_tile
    
    # Sort by size_pixels descending to prioritize larger areas
    df_sorted = df_tile.sort_values('size_pixels', ascending=False).reset_index(drop=False)
    
    coords = df_sorted[['centroid_y', 'centroid_x']].values
    original_indices = df_sorted['index'].tolist()
    keep_positions = []
    remaining_positions = list(range(len(df_sorted)))
    
    while remaining_positions:
        current_pos = remaining_positions[0]
        keep_positions.append(current_pos)
        
        current_coords = coords[current_pos:current_pos+1]
        
        distances = []
        for rem_pos in remaining_positions:
            dist = haversine_distance(
                current_coords[0, 0], current_coords[0, 1],
                coords[rem_pos, 0], coords[rem_pos, 1]
            )
            distances.append(dist)
        
        remaining_positions = [
            remaining_positions[i] 
            for i, dist in enumerate(distances) 
            if dist >= min_distance_km
        ]
    
    keep_original_indices = [original_indices[pos] for pos in keep_positions]
    return df_tile.loc[keep_original_indices]


deforest_df = pd.read_parquet(root_path / "data/interim/radd/south_america_combined.parquet")
len_before = len(deforest_df)
deforest_df = deforest_df[deforest_df['duration_days'] <= 30]
deforest_df = deforest_df[deforest_df['size_pixels'] >= 100]
deforest_df.reset_index(drop=True, inplace=True)
deforest_df = deforest_df[deforest_df['tile_name'] != '10S_050W_radd_alerts']

# Process each tile
cleaned_dfs = []
for tile_name in deforest_df['tile_name'].unique():
    df_tile = deforest_df[deforest_df['tile_name'] == tile_name].copy()
    df_filtered = filter_by_distance(df_tile, min_distance_km=1.5)
    cleaned_dfs.append(df_filtered)
    print(f"Tile {tile_name}: {len(df_tile)} -> {len(df_filtered)} samples")

# Combine all tiles back together
deforest_df_cleaned = pd.concat(cleaned_dfs, ignore_index=False)

# Store retained indices before any transformations
retained_positive_ids = deforest_df_cleaned.index.tolist()

# Convert to GeoDataFrame using centroid coordinates
gdf_cleaned = gpd.GeoDataFrame(
    deforest_df_cleaned, 
    geometry=gpd.points_from_xy(deforest_df_cleaned.centroid_x, deforest_df_cleaned.centroid_y),
    crs='EPSG:4326'
)

# Sort by index
gdf_cleaned = gdf_cleaned.sort_index()

# Write to parquet
gdf_cleaned.to_parquet(root_path / 'data/processed/radd/south_america_combined_clean_sampled_15.parquet')

print(f"\nTotal: {len_before} -> {len(gdf_cleaned)} samples")

print(f"Cleaned deforestation areas: {len(gdf_cleaned)} entries")