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

root_path = rootutils.find_root()

def initialize_gee():
    """Initialize Google Earth Engine"""
    print("🌍 Initializing Google Earth Engine...")
    try:
        ee.Initialize()
        print("✓ Google Earth Engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize GEE: {e}")
        print("Please authenticate with: earthengine authenticate")
        sys.exit(1)

def find_radd_tile_for_event(lat, lon, tile_boundaries_file="data/raw/radd/Deforestation_alerts_(RADD).geojson"):
    """
    Find which RADD tile contains the given lat/lon coordinates using actual tile boundaries
    
    Args:
        lat, lon: Event coordinates in EPSG:4326
        tile_boundaries_file: Path to RADD tile boundaries GeoJSON
        
    Returns:
        tile_id: String identifier like "10S_050W"
    """
    # Load tile boundaries (cache this globally for efficiency)
    if not hasattr(find_radd_tile_for_event, '_tiles_gdf'):
        tiles_gdf = gpd.read_file(tile_boundaries_file)
        # Ensure we know the CRS of the tile boundaries
        if tiles_gdf.crs is None:
            # Looking at the coordinates, this appears to be a projected system
            # You may need to set this based on the actual CRS of your data
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
    
    # If no tile found, return None or raise error
    print(f"Warning: No tile found for coordinates {lat}, {lon}")
    return None

def group_events_by_tile(deforest_df):
    """
    Group deforestation events by RADD tile
    
    Args:
        deforest_df: DataFrame with deforestation events
        
    Returns:
        tile_groups: Dict mapping tile_id to list of event indices
    """
    print("🗂️  Grouping events by RADD tile...")
    tile_groups = defaultdict(list)
    
    for idx, event in deforest_df.iterrows():
        tile_id = find_radd_tile_for_event(event['centroid_y'], event['centroid_x'])
        tile_groups[tile_id].append(idx)
    
    print(f"✓ Grouped {len(deforest_df)} events into {len(tile_groups)} tiles")
    for tile_id, event_indices in tile_groups.items():
        print(f"  {tile_id}: {len(event_indices)} events")
    
    return dict(tile_groups)

def load_radd_tile_data(tile_id, radd_base_dir="data/raw/radd/south_america"):
    """
    Load RADD raster data for a specific tile
    
    Args:
        tile_id: Tile identifier like "10S_050W"
        radd_base_dir: Base directory containing RADD tiles
        
    Returns:
        raster_data: numpy array with RADD alert data
        transform: Rasterio transform object
        crs: Coordinate reference system
    """
    print(f"📁 Loading RADD tile data for: {tile_id}")
    radd_file = Path(radd_base_dir) / f"{tile_id}_radd_alerts.tif"
    
    if not radd_file.exists():
        raise FileNotFoundError(f"RADD tile file not found: {radd_file}")
    
    with rasterio.open(radd_file) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs
    
    print(f"✓ Loaded RADD tile {tile_id}: {raster_data.shape}")
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
    """
    Extract RADD alerts for a specific region during the specified time period + safety buffer
    
    Args:
        radd_data: RADD raster data (encoded with dates)
        region_slice: Tuple of slices (row_slice, col_slice) defining the region
        earliest_alert: Start date of deforestation period
        latest_alert: End date of deforestation period
        safety_buffer_months: Additional months before/after to avoid (default: 2)
        
    Returns:
        temporal_mask: Boolean array indicating pixels with alerts in extended period
    """
    def date_to_radd_days(date_str):
        """Convert date string to RADD day encoding"""
        base_date = datetime(2014, 12, 31)
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        return (target_date - base_date).days
    
    # Add safety buffer to the time period
    earliest_date = datetime.strptime(earliest_alert, "%Y-%m-%d")
    latest_date = datetime.strptime(latest_alert, "%Y-%m-%d")
    
    # Extend period by safety buffer
    safety_buffer_days = safety_buffer_months * 30
    extended_start = earliest_date - timedelta(days=safety_buffer_days)
    extended_end = latest_date + timedelta(days=safety_buffer_days)
    
    start_days = date_to_radd_days(extended_start.strftime("%Y-%m-%d"))
    end_days = date_to_radd_days(extended_end.strftime("%Y-%m-%d"))
    
    # Extract only the region of interest
    row_slice, col_slice = region_slice
    region_data = radd_data[row_slice, col_slice]
    
    # Extract day component from RADD data
    valid_mask = region_data > 0
    day_component = region_data % 10000
    
    # Create temporal mask for alerts in the EXTENDED period
    temporal_mask = valid_mask & (day_component >= start_days) & (day_component <= end_days)
    
    return temporal_mask

def check_forest_cover_gee(lat, lon, date_str, forest_threshold=0.2):
    """
    Check if location has sufficient forest cover using Google Earth Engine
    
    Args:
        lat, lon: Coordinates to check
        date_str: Date for forest cover assessment (format: "YYYY-MM-DD")
        forest_threshold: Minimum forest fraction required
        
    Returns:
        is_forested: Boolean indicating if location meets forest criteria
        forest_fraction: Actual forest fraction at location
    """
    # Create point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # Use Dynamic World for forest cover
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Load Dynamic World data
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
           .filterBounds(point) \
           .filterDate(start_date, end_date) \
           .select('label')
    
    available_images = dw.size().getInfo()
    
    if available_images == 0:
        return False, 0.0
    
    # Get the most recent image
    latest_image = dw.sort('system:time_start', False).first()
    
    # Extract forest class (class 1 = trees/forest)
    forest_mask = latest_image.eq(1)
    
    # Calculate forest fraction in 1280m buffer around point
    buffer_size = 1280  # meters
    buffer_region = point.buffer(buffer_size)
    
    # Calculate forest fraction
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
                                     max_attempts=50, safety_buffer_months=2):
    """
    Sample a negative location for a single deforestation event
    
    Args:
        event_idx: Index of the event
        event: Event row from DataFrame
        radd_data: RADD raster data for the tile
        transform: Rasterio transform for the tile
        max_attempts: Maximum sampling attempts
        safety_buffer_months: Extra months before/after to avoid alerts
        
    Returns:
        negative_sample: Dict with negative sample info or None if failed
    """
    tile_height, tile_width = radd_data.shape
    region_size = 128  # 128x128 pixel region
    half_region = region_size // 2
    
    for attempt in range(max_attempts):
        # Sample random center point (ensuring 128x128 region fits within tile)
        center_row = random.randint(half_region, tile_height - half_region - 1)
        center_col = random.randint(half_region, tile_width - half_region - 1)
        
        # Define 128x128 region centered on this point
        row_start = center_row - half_region
        row_end = center_row + half_region
        col_start = center_col - half_region
        col_end = center_col + half_region
        
        region_slice = (slice(row_start, row_end), slice(col_start, col_end))
        
        # Extract temporal alerts for this region only
        temporal_mask = extract_temporal_alerts_region(
            radd_data, region_slice, 
            event['earliest_alert'], event['latest_alert'], 
            safety_buffer_months
        )
        
        # Check if center pixel has alerts during the extended time period
        center_in_region_row = half_region
        center_in_region_col = half_region
        
        if temporal_mask[center_in_region_row, center_in_region_col]:
            continue  # Skip - center has deforestation alerts in extended period
        
        # Convert center to coordinates
        sample_lat, sample_lon = pixel_to_coordinates(center_row, center_col, transform)
        
        # Check forest cover using GEE
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

def generate_negative_samples_tile_optimized(deforest_df, output_file=None, 
                                           radd_base_dir="data/raw/radd/south_america"):
    """
    Generate negative samples for all deforestation events using tile-first optimization
    
    Args:
        deforest_df: DataFrame with positive deforestation events
        output_file: Path to save negative samples CSV
        radd_base_dir: Directory containing RADD tiles
        
    Returns:
        negative_samples_df: DataFrame with negative samples
    """
    print("🚀 Starting optimized negative sample generation process")
    print(f"📊 Input: {len(deforest_df)} deforestation events")
    print(f"📂 RADD data directory: {radd_base_dir}")
    if output_file:
        print(f"💾 Output file: {output_file}")
    
    initialize_gee()
    
    # Group events by tile
    tile_groups = group_events_by_tile(deforest_df)
    
    negative_samples = []
    total_events_processed = 0
    
    print(f"\n🔄 Processing {len(tile_groups)} tiles")
    print("=" * 60)
    
    for tile_idx, (tile_id, event_indices) in enumerate(tile_groups.items()):
        print(f"\n📍 Processing tile {tile_idx + 1}/{len(tile_groups)}: {tile_id}")
        print(f"  Events in this tile: {len(event_indices)}")
        
        # Load tile data
        try:
            radd_data, transform, crs = load_radd_tile_data(tile_id, radd_base_dir)
            print(f"  ✓ Tile loaded: {radd_data.shape}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            print(f"  Skipping all {len(event_indices)} events in this tile")
            total_events_processed += len(event_indices)
            continue
        
        # Process each event in this tile
        tile_successful_samples = 0
        for event_idx in event_indices:
            event = deforest_df.loc[event_idx]
            total_events_processed += 1
            
            print(f"    Event {total_events_processed}/{len(deforest_df)} (ID {event_idx}): "
                  f"{event['centroid_y']:.4f}, {event['centroid_x']:.4f}")
            
            # Sample negative location
            negative_sample = sample_negative_location_for_event(
                event_idx, event, radd_data, transform
            )
            
            if negative_sample:
                negative_sample['tile_id'] = tile_id
                negative_samples.append(negative_sample)
                tile_successful_samples += 1
                print(f"      ✅ Success (attempt {negative_sample['attempt_number']})")
            else:
                print(f"      ❌ Failed after 50 attempts")
        
        # Tile summary
        tile_success_rate = tile_successful_samples / len(event_indices) * 100 if event_indices else 0
        print(f"  📊 Tile {tile_id} complete: {tile_successful_samples}/{len(event_indices)} "
              f"samples ({tile_success_rate:.1f}% success)")
        
        # Overall progress update
        overall_success_rate = len(negative_samples) / total_events_processed * 100
        print(f"  🏃 Overall progress: {total_events_processed}/{len(deforest_df)} events, "
              f"{len(negative_samples)} samples ({overall_success_rate:.1f}% success)")
    
    # Convert to DataFrame
    print(f"\n📋 Creating final DataFrame...")
    negative_samples_df = pd.DataFrame(negative_samples)
    
    if len(negative_samples_df) > 0:
        print(f"\n🎉 Negative sampling completed successfully!")
        print(f"✓ Generated {len(negative_samples_df)} negative samples")
        print(f"✓ Overall success rate: {len(negative_samples_df) / len(deforest_df) * 100:.1f}%")
        print(f"✓ Processed {len(tile_groups)} unique RADD tiles")
        
        # Display sample statistics
        if 'forest_fraction' in negative_samples_df.columns:
            print(f"✓ Average forest fraction: {negative_samples_df['forest_fraction'].mean():.3f}")
        
        # Save if output file specified
        if output_file:
            print(f"\n💾 Saving results to file...")
            negative_samples_df.to_csv(output_file, index=False)
            print(f"✓ Saved negative samples to: {output_file}")
    else:
        print("\n❌ No negative samples generated")
        print("   Check RADD data availability and forest cover thresholds")
    
    return negative_samples_df

if __name__ == "__main__":
    print("🔧 Loading deforestation events dataset...")
    # Load deforestation events
    deforest_df = pd.read_parquet(root_path / "data/processed/radd/south_america_combined_clean.parquet")
    print(f"✓ Loaded {len(deforest_df)} deforestation events")

    # Generate negative samples
    negative_samples_df = generate_negative_samples_tile_optimized(
        deforest_df.head(50),
        output_file=root_path / "data/processed/radd/negative_samples.csv"
    )

    print("\n🏁 Optimized negative sampling process complete!")
    if len(negative_samples_df) > 0:
        print(f"Final results saved to: {root_path / 'data/processed/radd/negative_samples.csv'}")
    else:
        print("No samples were successfully generated. Check logs above for issues.")