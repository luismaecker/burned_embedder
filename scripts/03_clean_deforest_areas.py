import pandas as pd
import rootutils

root_path = rootutils.find_root()

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth in km
    """
    R = 6371
    
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


def process_continent_data(input_path, output_path, min_distance_km=1.5, 
                          max_duration_days=30, min_size_pixels=100, 
                          excluded_tiles=None):
    """
    Process a single continent's parquet file with filtering and spatial sampling.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save cleaned output
        min_distance_km: Minimum distance between samples in km
        max_duration_days: Maximum duration to keep
        min_size_pixels: Minimum size in pixels to keep
        excluded_tiles: List of tile names to exclude (optional)
    """
    deforest_df = pd.read_parquet(input_path)
    len_before = len(deforest_df)
    
    # Show filter impact step by step
    print(f"  Initial samples: {len(deforest_df)}")
    
    # Duration filter
    deforest_df = deforest_df[deforest_df['duration_days'] <= max_duration_days]
    print(f"  After duration <= {max_duration_days}: {len(deforest_df)}")
    
    # Size filter
    deforest_df = deforest_df[deforest_df['size_pixels'] >= min_size_pixels]
    print(f"  After size >= {min_size_pixels}: {len(deforest_df)}")
    
    deforest_df.reset_index(drop=True, inplace=True)
    
    # Exclude specific tiles if provided
    if excluded_tiles:
        for tile in excluded_tiles:
            deforest_df = deforest_df[deforest_df['tile_name'] != tile]
        print(f"  After excluding tiles: {len(deforest_df)}")
    
    # Process each tile with distance filtering
    cleaned_dfs = []
    for tile_name in deforest_df['tile_name'].unique():
        df_tile = deforest_df[deforest_df['tile_name'] == tile_name].copy()
        df_filtered = filter_by_distance(df_tile, min_distance_km=min_distance_km)
        cleaned_dfs.append(df_filtered)
        print(f"    Tile {tile_name}: {len(df_tile)} -> {len(df_filtered)} samples")
    
    # Combine all tiles
    deforest_df_cleaned = pd.concat(cleaned_dfs, ignore_index=False)
    
    # Convert to GeoDataFrame
    gdf_cleaned = gpd.GeoDataFrame(
        deforest_df_cleaned, 
        geometry=gpd.points_from_xy(deforest_df_cleaned.centroid_x, deforest_df_cleaned.centroid_y),
        crs='EPSG:4326'
    )
    
    # Sort by index
    gdf_cleaned = gdf_cleaned.sort_index()
    
    # Save to parquet
    gdf_cleaned.to_parquet(output_path)
    
    print(f"  Final total: {len_before} -> {len(gdf_cleaned)} samples")
    print(f"  Reduction: {(1 - len(gdf_cleaned)/len_before)*100:.1f}%\n")
    
    return gdf_cleaned


def process_all_continents(min_distance_km=1.5, max_duration_days=30, 
                          min_size_pixels=100):
    """
    Process all three continent parquet files.
    """
    continents = {
        'south_america': {
            'input': root_path / "data/interim/radd/south_america_combined.parquet",
            'output': root_path / "data/processed/radd/south_america_combined_clean.parquet",
            'excluded_tiles': ['10S_050W_radd_alerts']
        },
        'africa': {
            'input': root_path / "data/interim/radd/africa_combined.parquet",
            'output': root_path / "data/processed/radd/africa_combined_clean.parquet",
            'excluded_tiles': None
        },
        'southeast_asia': {
            'input': root_path / "data/interim/radd/southeast_asia_combined.parquet",
            'output': root_path / "data/processed/radd/southeast_asia_combined_clean.parquet",
            'excluded_tiles': None
        }
    }
    
    results = {}
    
    for continent_name, paths in continents.items():
        print(f"Processing {continent_name}...")
        
        # Create output directory if needed
        paths['output'].parent.mkdir(parents=True, exist_ok=True)
        
        # Process the continent
        gdf_cleaned = process_continent_data(
            input_path=paths['input'],
            output_path=paths['output'],
            min_distance_km=min_distance_km,
            max_duration_days=max_duration_days,
            min_size_pixels=min_size_pixels,
            excluded_tiles=paths['excluded_tiles']
        )
        
        results[continent_name] = gdf_cleaned
        print(f"Saved {continent_name} to {paths['output']}\n")
    
    return results


if __name__ == "__main__":
    # Show some data stats first
    print("Analyzing data distributions...\n")
    
    for continent in ['south_america', 'africa', 'southeast_asia']:
        parquet_path = root_path / f"data/interim/radd/{continent}_combined.parquet"
        df = pd.read_parquet(parquet_path)
        print(f"{continent}:")
        print(f"  Duration stats: min={df['duration_days'].min()}, max={df['duration_days'].max()}, median={df['duration_days'].median()}")
        print(f"  Size stats: min={df['size_pixels'].min()}, max={df['size_pixels'].max()}, median={df['size_pixels'].median()}")
        print(f"  Duration <= 30: {(df['duration_days'] <= 30).sum()} / {len(df)} ({(df['duration_days'] <= 30).mean()*100:.1f}%)")
        print(f"  Size >= 100: {(df['size_pixels'] >= 100).sum()} / {len(df)} ({(df['size_pixels'] >= 100).mean()*100:.1f}%)")
        print()
    
    print("="*60)
    print("Starting processing...\n")
    
    results = process_all_continents(
        min_distance_km=1.5,
        max_duration_days=30,
        min_size_pixels=50
    )
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for continent, gdf in results.items():
        print(f"{continent.replace('_', ' ').title()}: {len(gdf)} samples")