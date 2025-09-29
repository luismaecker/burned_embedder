import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import random
from pathlib import Path
import ee
from datetime import datetime, timedelta
import sys
import rootutils
from collections import defaultdict
from tqdm import tqdm

root_path = rootutils.find_root()

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        print("Please authenticate with: earthengine authenticate")
        sys.exit(1)

def find_radd_tile_for_event(lat, lon, tile_boundaries_file="data/raw/radd/Deforestation_alerts_(RADD).geojson"):
    """Find which RADD tile contains the given lat/lon coordinates using actual tile boundaries"""
    # Load tile boundaries (cache this globally for efficiency)
    if not hasattr(find_radd_tile_for_event, '_tiles_gdf'):
        tiles_gdf = gpd.read_file(tile_boundaries_file)
        if tiles_gdf.crs is None:
            print("Warning: Tile boundaries have no CRS defined")
        find_radd_tile_for_event._tiles_gdf = tiles_gdf
    
    tiles_gdf = find_radd_tile_for_event._tiles_gdf
    
    # Create point in lat/lon (EPSG:4326)
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
    
    # Transform point to same CRS as tile boundaries
    if tiles_gdf.crs != point_gdf.crs:
        point_gdf = point_gdf.to_crs(tiles_gdf.crs)
    
    # Find intersecting tile
    point_geom = point_gdf.geometry.iloc[0]
    for idx, tile in tiles_gdf.iterrows():
        if tile.geometry.contains(point_geom):
            return tile['tile_id']
    
    print(f"Warning: No tile found for coordinates {lat}, {lon}")
    return None

def group_events_by_tile(deforest_df):
    """Group deforestation events by RADD tile"""
    tile_groups = defaultdict(list)
    
    for idx, event in deforest_df.iterrows():
        tile_id = find_radd_tile_for_event(event['centroid_y'], event['centroid_x'])
        if tile_id:
            tile_groups[tile_id].append(idx)
    
    return dict(tile_groups)

def load_radd_tile_data(tile_id, radd_base_dir="data/raw/radd/south_america"):
    """Load RADD raster data for a specific tile"""
    radd_file = Path(radd_base_dir) / f"{tile_id}_radd_alerts.tif"
    
    if not radd_file.exists():
        raise FileNotFoundError(f"RADD tile file not found: {radd_file}")
    
    with rasterio.open(radd_file) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs
    
    return raster_data, transform, crs

def pixel_to_coordinates(row, col, transform):
    """Convert pixel coordinates to lat/lon"""
    x, y = rasterio.transform.xy(transform, row, col)
    return y, x  # lat, lon

def coordinates_to_pixel(lat, lon, transform):
    """Convert lat/lon to pixel coordinates"""
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return row, col

def extract_temporal_alerts_region(radd_data, region_slice, earliest_alert, latest_alert, safety_buffer_months=2):
    """Extract RADD alerts for a specific region during the specified time period + safety buffer"""
    def date_to_radd_days(date_str):
        base_date = datetime(2014, 12, 31)
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        return (target_date - base_date).days
    
    earliest_date = datetime.strptime(earliest_alert, "%Y-%m-%d")
    latest_date = datetime.strptime(latest_alert, "%Y-%m-%d")
    
    safety_buffer_days = safety_buffer_months * 30
    extended_start = earliest_date - timedelta(days=safety_buffer_days)
    extended_end = latest_date + timedelta(days=safety_buffer_days)
    
    start_days = date_to_radd_days(extended_start.strftime("%Y-%m-%d"))
    end_days = date_to_radd_days(extended_end.strftime("%Y-%m-%d"))
    
    row_slice, col_slice = region_slice
    region_data = radd_data[row_slice, col_slice]
    
    valid_mask = region_data > 0
    day_component = region_data % 10000
    
    temporal_mask = valid_mask & (day_component >= start_days) & (day_component <= end_days)
    
    return temporal_mask

def check_forest_cover_gee(lat, lon, date_str, forest_threshold=0.2):
    """Check if location has sufficient forest cover using Google Earth Engine"""
    point = ee.Geometry.Point([lon, lat])
    
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=30)).strftime("%Y-%m-%d")
    
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
           .filterBounds(point) \
           .filterDate(start_date, end_date) \
           .select('label')
    
    available_images = dw.size().getInfo()
    
    if available_images == 0:
        return False, 0.0
    
    latest_image = dw.sort('system:time_start', False).first()
    forest_mask = latest_image.eq(1)
    
    buffer_size = 1280  # meters
    buffer_region = point.buffer(buffer_size)
    
    forest_stats = forest_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_region,
        scale=10,
        maxPixels=1e6
    )
    
    forest_fraction = forest_stats.getInfo().get('label', 0)
    is_forested = forest_fraction >= forest_threshold
    
    return is_forested, forest_fraction

def sample_negative_location_for_event(event_idx, event, radd_data, transform, 
                                     max_attempts=20, safety_buffer_months=2):
    """Sample a negative location for a single deforestation event"""
    tile_height, tile_width = radd_data.shape
    region_size = 128
    half_region = region_size // 2
    
    for attempt in range(max_attempts):
        center_row = random.randint(half_region, tile_height - half_region - 1)
        center_col = random.randint(half_region, tile_width - half_region - 1)
        
        row_start = center_row - half_region
        row_end = center_row + half_region
        col_start = center_col - half_region
        col_end = center_col + half_region
        
        region_slice = (slice(row_start, row_end), slice(col_start, col_end))
        
        temporal_mask = extract_temporal_alerts_region(
            radd_data, region_slice, 
            event['earliest_alert'], event['latest_alert'], 
            safety_buffer_months
        )
        
        center_in_region_row = half_region
        center_in_region_col = half_region
        
        if temporal_mask[center_in_region_row, center_in_region_col]:
            continue
        
        sample_lat, sample_lon = pixel_to_coordinates(center_row, center_col, transform)
        
        try:
            is_forested, forest_fraction = check_forest_cover_gee(
                sample_lat, sample_lon, event['earliest_alert'], forest_threshold=0.2
            )
            
            if is_forested:
                return {
                    'lat': sample_lat,
                    'lon': sample_lon,
                    'forest_fraction': forest_fraction,
                    'earliest_alert': event['earliest_alert'],
                    'latest_alert': event['latest_alert'],
                    'safety_buffer_months': safety_buffer_months,
                    'is_deforestation': False,
                    'attempt_number': attempt + 1,
                    'positive_event_id': event_idx,
                    'positive_lat': event['centroid_y'],
                    'positive_lon': event['centroid_x'],
                    'positive_area_hectares': event.get('area_hectares', np.nan),
                    'positive_duration_days': event.get('duration_days', np.nan)
                }
            
        except Exception as e:
            continue
    
    return None

def generate_negative_samples(deforest_df, output_file=None, 
                             radd_base_dir="data/raw/radd/south_america"):
    """
    Generate negative samples for all deforestation events processing tile by tile
    
    Args:
        deforest_df: DataFrame with positive deforestation events
        output_file: Path to save negative samples parquet
        radd_base_dir: Directory containing RADD tiles
        
    Returns:
        negative_samples_df: DataFrame with negative samples
    """
    print(f"Starting negative sample generation for {len(deforest_df)} events")
    
    # Initialize GEE once
    initialize_gee()
    
    # Group events by tile
    tile_groups = group_events_by_tile(deforest_df)
    print(f"Grouped events into {len(tile_groups)} tiles")
    
    negative_samples = []
    
    # Process each tile
    for tile_idx, (tile_id, event_indices) in enumerate(tile_groups.items()):
        print(f"\nProcessing tile {tile_idx + 1}/{len(tile_groups)}: {tile_id} ({len(event_indices)} events)")
        
        try:
            # Load tile data
            radd_data, transform, crs = load_radd_tile_data(tile_id, radd_base_dir)
            
            # Process each event in this tile with progress bar
            for event_idx in tqdm(event_indices, desc=f"Events in {tile_id}", leave=False):
                event = deforest_df.loc[event_idx]
                
                negative_sample = sample_negative_location_for_event(
                    event_idx, event, radd_data, transform
                )
                
                if negative_sample:
                    negative_sample['tile_id'] = tile_id
                    negative_samples.append(negative_sample)
        
        except FileNotFoundError:
            print(f"  Skipping tile {tile_id}: file not found")
            continue
        except Exception as e:
            print(f"  Error processing tile {tile_id}: {e}")
            continue
    
    negative_samples_df = pd.DataFrame(negative_samples)
    
    if len(negative_samples_df) > 0:
        # Convert to GeoDataFrame with EPSG:4326
        geometry = [Point(row['lon'], row['lat']) for _, row in negative_samples_df.iterrows()]
        negative_samples_gdf = gpd.GeoDataFrame(negative_samples_df, geometry=geometry, crs='EPSG:4326')
        
        print(f"\nGenerated {len(negative_samples_df)} negative samples")
        print(f"Success rate: {len(negative_samples_df) / len(deforest_df) * 100:.1f}%")
        
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix.lower() != '.parquet':
                output_path = output_path.with_suffix('.parquet')
            
            negative_samples_gdf.to_parquet(output_path)
            print(f"Saved to: {output_path}")
    else:
        print("No negative samples generated")
    
    return negative_samples_df

if __name__ == '__main__':
    # Load deforestation events
    deforest_df = pd.read_parquet(root_path / "data/processed/radd/south_america_combined_clean_sampled.parquet")
    print(f"Loaded {len(deforest_df)} deforestation events")
    
    # Generate negative samples sequentially
    negative_samples_df = generate_negative_samples(
        deforest_df,
        output_file=root_path / "data/processed/radd/negative_samples_sequential.parquet"
    )
    
    print("Sequential negative sampling complete!")