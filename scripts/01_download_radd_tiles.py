import os
import pandas as pd
import geopandas as gpd
import requests
from pathlib import Path
import time

def download_radd_tiles(geojson_file_path, tile_configs, base_dir="data/raw/radd"):
    """
    Download RADD tiles to specified folders
    
    tile_configs format:
    {
        "folder_name": ["tile_id1", "tile_id2", ...],
        "another_folder": ["tile_id3", "tile_id4", ...]
    }
    """
    # Load data
    df = gpd.read_file(geojson_file_path)
    print(f"Loaded {len(df)} tiles from CSV")
    
    # Download tiles for each folder
    for folder_name, tile_ids in tile_configs.items():
        print(f"\nDownloading {len(tile_ids)} tiles to {folder_name}")
        
        # Create folder
        folder_path = Path(base_dir) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Filter for requested tiles
        tiles_to_download = df[df['tile_id'].isin(tile_ids)]
        
        # Check for missing tiles
        found_tiles = set(tiles_to_download['tile_id'].tolist())
        requested_tiles = set(tile_ids)
        missing_tiles = requested_tiles - found_tiles
        
        if missing_tiles:
            print(f"Warning: These tiles not found: {missing_tiles}")
        
        # Download each tile
        for idx, row in tiles_to_download.iterrows():
            tile_id = row['tile_id']
            download_url = row['download']
            
            filename = f"{tile_id}_radd_alerts.tif"
            filepath = folder_path / filename
            
            # Skip if exists
            if filepath.exists():
                print(f"  Skipping {tile_id} - already exists")
                continue
            
            print(f"  Downloading {tile_id}")
            
            try:
                response = requests.get(download_url, timeout=60)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"    ✓ Downloaded {filename}")
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"    ✗ Failed: {e}")

def list_available_tiles(geojson_file_path):
    """Show all available tile_ids"""
    df = gpd.read_file(geojson_file_path)
    return df['tile_id'].tolist()

if __name__ == "__main__":
    geojson_path = "data/raw/radd/Deforestation_alerts_(RADD).geojson"  # Can be downloaded from https://data.globalforestwatch.org/datasets/gfw::deforestation-alerts-radd/about
    
    # Uncomment to see available tiles
    tiles = list_available_tiles(geojson_path)
    print(f"\nTotal available tiles: {len(tiles)}")
    print(tiles)

    tile_configs = {
        "all": tiles  # Download all tiles into one folder
    }

    # Configure which tiles go to which folders
    tile_configs = {
        "south_america": [
            "10S_060W", "10S_070W",
            "20S_060W", "20S_070W",
            "00N_050W", "00N_060W", "00N_070W", "00N_080W",
            "10N_060W", "10N_070W", "10N_080W"
        ],
        # "africa": [
        #     "00N_020E", "00N_010E"
        # ],
        # "southeast_asia": [
        #     "10N_100E", "10N_110E",
        #     "00N_130E", "00N_140E"
        # ]
    }
    
    # Download the tiles
    download_radd_tiles(geojson_path, tile_configs)
    print("\nAll downloads complete!")