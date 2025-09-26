import numpy as np
import rasterio
from skimage.measure import label
import time
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm


def days_to_date(days_since_dec31_2014):
    """Convert days since Dec 31, 2014 to actual date"""
    base_date = datetime(2014, 12, 31)
    return base_date + timedelta(days=int(days_since_dec31_2014))

def process_raster_subset_simple(input_path, min_area_pixels, confidence_level='high',
                                subset_size=5000, output_prefix="test_subset"):
    """
    Ultra-simple version: process only a small subset of the raster for testing.
    No dask, no multiprocessing, just basic numpy operations.
    """
    
    print(f"=== SIMPLE SUBSET TEST ===")
    print(f"Processing {subset_size}x{subset_size} subset from top-left corner")
    print(f"Confidence level: {confidence_level}")
    print(f"Min area threshold: {min_area_pixels} pixels")
    
    start_time = time.time()
    
    # Step 1: Load just a subset of the raster
    print("1. Loading subset of raster data...")
    with rasterio.open(input_path) as src:
        # Read only the top-left subset
        subset_data = src.read(1, window=rasterio.windows.Window(0, 0, subset_size, subset_size))
        
        # Get spatial reference info
        raster_transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        
        print(f"Subset shape: {subset_data.shape}")
        print(f"Data type: {subset_data.dtype}")
    
    # Step 2: Decode alert data
    print("2. Decoding alert data...")
    
    # Create masks
    valid_mask = subset_data > 0
    first_digit = subset_data // 10000
    
    if confidence_level == 'high':
        analysis_mask = (first_digit == 3) & valid_mask
    elif confidence_level == 'low':
        analysis_mask = (first_digit == 2) & valid_mask
    else:
        analysis_mask = valid_mask
    
    # Basic statistics
    total_pixels = subset_data.size
    valid_pixels = np.sum(valid_mask)
    analyzed_pixels = np.sum(analysis_mask)
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
    print(f"Analysis pixels: {analyzed_pixels:,} ({analyzed_pixels/total_pixels*100:.2f}%)")
    
    if analyzed_pixels == 0:
        print("No pixels to analyze in this subset!")
        return None
    
    # Step 3: Find connected components (simple, single-threaded)
    print("3. Finding connected components...")
    cc_start = time.time()
    
    # Simple connected components
    labeled_array = label(analysis_mask.astype(np.uint8), connectivity=2)
    unique_labels = np.unique(labeled_array[labeled_array > 0])
    
    cc_time = time.time() - cc_start
    print(f"Found {len(unique_labels)} components in {cc_time:.2f} seconds")
    
    # Step 4: Calculate component properties
    print("4. Calculating component properties...")
    
    large_components = []
    component_features = []
    
    pixel_area_deg2 = abs(raster_transform[0] * raster_transform[4])
    area_conversion = pixel_area_deg2 * 1232100  # Very rough hectare conversion
    
    for label_id in tqdm(unique_labels, desc="Processing components"):
        component_mask = labeled_array == label_id
        component_size = np.sum(component_mask)

        if component_size >= min_area_pixels:

            # Get component pixels
            pixel_coords = np.where(component_mask)
            pixel_rows, pixel_cols = pixel_coords
            
            # Calculate centroid
            centroid_row = np.mean(pixel_rows)
            centroid_col = np.mean(pixel_cols)
            
            # Convert to geographic coordinates
            centroid_x, centroid_y = raster_transform * (centroid_col, centroid_row)
            
            # Get temporal information (sample up to 50 pixels)
            sample_size = min(50, len(pixel_rows))
            sample_indices = np.random.choice(len(pixel_rows), sample_size, replace=False)
            
            component_days = []
            for idx in sample_indices:
                row, col = pixel_rows[idx], pixel_cols[idx]
                value = subset_data[row, col]
                if value > 0:
                    component_days.append(value % 10000)
            
            # Calculate temporal stats
            if component_days:
                min_days = min(component_days)
                max_days = max(component_days)
                earliest_date = days_to_date(min_days)
                latest_date = days_to_date(max_days)
                duration_days = max_days - min_days
            else:
                earliest_date = latest_date = datetime(2015, 1, 1)
                duration_days = 0
            
            # Calculate area
            area_hectares = component_size * area_conversion
            
            large_components.append(label_id)
            
            # Create feature
            feature_attrs = {
                'component_id': int(label_id),
                'size_pixels': int(component_size),
                'area_hectares': round(area_hectares, 6),
                'earliest_alert': earliest_date.strftime('%Y-%m-%d'),
                'latest_alert': latest_date.strftime('%Y-%m-%d'),
                'duration_days': int(duration_days),
                'centroid_x': round(centroid_x, 6),
                'centroid_y': round(centroid_y, 6),
                'confidence': confidence_level
            }
            
            component_features.append({
                'geometry': Point(centroid_x, centroid_y),
                'properties': feature_attrs
            })
    
    # Step 5: Create outputs
    print("5. Creating outputs...")
    
    # Create filtered raster
    filtered_raster = np.zeros_like(labeled_array, dtype=np.uint16)
    for comp_id in large_components:
        filtered_raster[labeled_array == comp_id] = comp_id
    
    # Save subset raster
    raster_output = f"{output_prefix}_subset_raster.tif"
    
    # Update profile for subset
    profile.update({
        'width': subset_size,
        'height': subset_size,
        'transform': raster_transform
    })
    
    with rasterio.open(raster_output, 'w', **profile) as dst:
        dst.write(filtered_raster, 1)
    
    print(f"Saved subset raster: {raster_output}")
    
    # Save vector data
    if component_features:
        centroids_gdf = gpd.GeoDataFrame([
            {'geometry': f['geometry'], **f['properties']} 
            for f in component_features
        ], crs=crs)
        
        # Save formats
        geojson_output = f"{output_prefix}_centroids.geojson"
        csv_output = f"{output_prefix}_centroids.csv"
        
        centroids_gdf.to_file(geojson_output, driver='GeoJSON')
        centroids_gdf.drop('geometry', axis=1).to_csv(csv_output, index=False)
        
        print(f"Saved vector data:")
        print(f"  GeoJSON: {geojson_output}")
        print(f"  CSV: {csv_output}")
        
        # Show results
        print(f"\nResults summary:")
        print(f"Components found: {len(unique_labels):,}")
        print(f"Large components: {len(large_components):,}")
        
        if component_features:
            areas = [f['properties']['area_hectares'] for f in component_features]
            print(f"Area range: {min(areas):.4f} to {max(areas):.4f} hectares")
            
            print(f"\nSample data:")
            print(centroids_gdf[['component_id', 'size_pixels', 'area_hectares', 
                               'earliest_alert', 'duration_days']].head())
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.1f} seconds")
    
    return {
        'raster_output': raster_output,
        'vector_outputs': {'geojson': geojson_output, 'csv': csv_output} if component_features else {},
        'n_components': len(large_components),
        'total_area_hectares': sum(f['properties']['area_hectares'] for f in component_features)
    }

# Main execution - SIMPLE TEST VERSION
if __name__ == "__main__":
    INPUT_RASTER = "data/raw/10N_080W.tif"
    MIN_AREA_PIXELS = 100      # Lower threshold for testing
    CONFIDENCE_LEVEL = 'high'
    OUTPUT_PREFIX = "simple_test"
    SUBSET_SIZE = 10000        # Small subset for testing
    
    try:
        print("Testing with small subset first...")
        
        results = process_raster_subset_simple(
            input_path=INPUT_RASTER,
            min_area_pixels=MIN_AREA_PIXELS,
            confidence_level=CONFIDENCE_LEVEL,
            subset_size=SUBSET_SIZE,
            output_prefix=OUTPUT_PREFIX
        )
        
        if results:
            print(f"\n=== TEST SUCCESSFUL ===")
            print(f"Found {results['n_components']} large components")
            print(f"Total area: {results['total_area_hectares']:.2f} hectares")
            print(f"Files created:")
            print(f"  {results['raster_output']}")
            for fmt, path in results['vector_outputs'].items():
                print(f"  {path}")
            
            print(f"\nIf this looks good, we can scale up to the full raster!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()