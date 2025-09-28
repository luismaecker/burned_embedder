import numpy as np
import rasterio
from skimage.measure import label
import time
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import random
from multiprocessing import Pool
import functools
import os
import glob
from pathlib import Path


def days_to_date(days_since_dec31_2014):
    """Convert days since Dec 31, 2014 to actual date"""
    base_date = datetime(2014, 12, 31)
    return base_date + timedelta(days=int(days_since_dec31_2014))

def get_tif_files(folder_path):
    """Get all .tif files in a folder"""
    return sorted(glob.glob(os.path.join(folder_path, "*.tif")) + 
                  glob.glob(os.path.join(folder_path, "*.tiff")))

def get_random_sampling_windows(raster_path, n_samples=10, window_size=10000, seed=42):
    """Generate random sampling windows"""
    random.seed(seed)
    np.random.seed(seed)
    
    with rasterio.open(raster_path) as src:
        raster_height, raster_width = src.height, src.width
        
    max_row = raster_height - window_size
    max_col = raster_width - window_size
    
    windows = []
    for i in range(n_samples):
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        window = rasterio.windows.Window(start_col, start_row, window_size, window_size)
        windows.append({
            'window': window,
            'id': f"sample_{i+1:02d}",
            'start_row': start_row,
            'start_col': start_col
        })
    
    return windows

def process_single_sample(window_info, input_path, min_area_pixels, confidence_level,
                         output_folder, output_prefix, tile_name):
    """Process a single sample window"""
    
    sample_id = window_info['id']
    window = window_info['window']
    start_row = window_info['start_row']
    start_col = window_info['start_col']
    
    try:
        # Load data
        with rasterio.open(input_path) as src:
            sample_data = src.read(1, window=window)
            raster_transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
            window_transform = rasterio.windows.transform(window, raster_transform)
        
        # Decode alert data
        valid_mask = sample_data > 0
        first_digit = sample_data // 10000
        
        if confidence_level == 'high':
            analysis_mask = (first_digit == 3) & valid_mask
        elif confidence_level == 'low':
            analysis_mask = (first_digit == 2) & valid_mask
        else:
            analysis_mask = valid_mask
        
        if np.sum(analysis_mask) == 0:
            return None
        
        # Find connected components
        labeled_array = label(analysis_mask.astype(np.uint8), connectivity=2)
        unique_labels = np.unique(labeled_array[labeled_array > 0])
        
        # Process components
        large_components = []
        component_features = []
        pixel_area_deg2 = abs(window_transform[0] * window_transform[4])
        area_conversion = pixel_area_deg2 * 1232100
        
        for label_id in unique_labels:
            component_mask = labeled_array == label_id
            component_size = np.sum(component_mask)

            if component_size >= min_area_pixels:
                pixel_coords = np.where(component_mask)
                pixel_rows, pixel_cols = pixel_coords
                
                centroid_row = np.mean(pixel_rows)
                centroid_col = np.mean(pixel_cols)
                centroid_x, centroid_y = window_transform * (centroid_col, centroid_row)
                
                # Sample temporal data
                sample_size = min(50, len(pixel_rows))
                sample_indices = np.random.choice(len(pixel_rows), sample_size, replace=False)
                
                component_days = []
                for idx in sample_indices:
                    row, col = pixel_rows[idx], pixel_cols[idx]
                    value = sample_data[row, col]
                    if value > 0:
                        component_days.append(value % 10000)
                
                if component_days:
                    min_days = min(component_days)
                    max_days = max(component_days)
                    earliest_date = days_to_date(min_days)
                    latest_date = days_to_date(max_days)
                    duration_days = max_days - min_days
                else:
                    earliest_date = latest_date = datetime(2015, 1, 1)
                    duration_days = 0
                
                area_hectares = component_size * area_conversion
                large_components.append(label_id)
                
                feature_attrs = {
                    'tile_name': tile_name,
                    'sample_id': sample_id,
                    'component_id': int(label_id),
                    'size_pixels': int(component_size),
                    'area_hectares': round(area_hectares, 6),
                    'earliest_alert': earliest_date.strftime('%Y-%m-%d'),
                    'latest_alert': latest_date.strftime('%Y-%m-%d'),
                    'duration_days': int(duration_days),
                    'centroid_x': round(centroid_x, 6),
                    'centroid_y': round(centroid_y, 6),
                    'confidence': confidence_level,
                    'date_raster_file': f"{sample_id}_dates.tif"
                }
                
                component_features.append({
                    'geometry': Point(centroid_x, centroid_y),
                    'properties': feature_attrs
                })
        
        # Save outputs
        # Component raster
        filtered_raster = np.zeros_like(labeled_array, dtype=np.uint16)
        for comp_id in large_components:
            filtered_raster[labeled_array == comp_id] = comp_id
        
        raster_output = os.path.join(output_folder, f"{sample_id}_components.tif")
        profile.update({
            'width': sample_data.shape[1],
            'height': sample_data.shape[0],
            'transform': window_transform
        })
        
        with rasterio.open(raster_output, 'w', **profile) as dst:
            dst.write(filtered_raster, 1)
        
        # Date raster
        date_raster = np.zeros_like(sample_data, dtype=np.int16)
        date_values = sample_data % 10000
        date_raster[valid_mask] = date_values[valid_mask]
        
        date_raster_output = os.path.join(output_folder, f"{sample_id}_dates.tif")
        date_profile = profile.copy()
        date_profile.update({'dtype': 'int16', 'nodata': 0, 'compress': 'lzw'})
        
        with rasterio.open(date_raster_output, 'w', **date_profile) as dst:
            dst.write(date_raster, 1)
        
        # Temporary vector files
        temp_files = []
        if component_features:
            centroids_gdf = gpd.GeoDataFrame([
                {'geometry': f['geometry'], **f['properties']} 
                for f in component_features
            ], crs=crs)
            
            geojson_output = os.path.join(output_folder, f"{sample_id}_centroids.geojson")
            csv_output = os.path.join(output_folder, f"{sample_id}_centroids.csv")
            
            centroids_gdf.to_file(geojson_output, driver='GeoJSON')
            centroids_gdf.drop('geometry', axis=1).to_csv(csv_output, index=False)
            temp_files = [geojson_output, csv_output]
        
        print(f"[PID {os.getpid()}] Completed {sample_id} from {tile_name}")
        
        return {
            'sample_id': sample_id,
            'tile_name': tile_name,
            'temp_files': temp_files,
            'n_components': len(large_components),
            'total_area_hectares': sum(f['properties']['area_hectares'] for f in component_features) if component_features else 0,
            'component_features': component_features
        }
        
    except Exception as e:
        print(f"[PID {os.getpid()}] Failed {sample_id} from {tile_name}: {e}")
        return None

def process_random_samples_parallel(input_path, min_area_pixels, confidence_level,
                                  n_samples, window_size, output_folder, seed, n_processes):
    """Process multiple samples in parallel"""
    
    # Setup - create tile-specific folder
    tile_name = Path(input_path).stem
    tile_folder = os.path.join(output_folder, tile_name)
    output_prefix = tile_name
    Path(tile_folder).mkdir(parents=True, exist_ok=True)
    
    if n_processes is None:
        n_processes = min(n_samples, os.cpu_count())
    
    print(f"Processing {tile_name}: {n_samples} samples using {n_processes} processes")
    
    # Generate windows and process
    sample_windows = get_random_sampling_windows(input_path, n_samples, window_size, seed)
    
    process_func = functools.partial(
        process_single_sample,
        input_path=input_path,
        min_area_pixels=min_area_pixels,
        confidence_level=confidence_level,
        output_folder=tile_folder,
        output_prefix=output_prefix,
        tile_name=tile_name
    )
    
    with Pool(processes=n_processes) as pool:
        all_results = pool.map(process_func, sample_windows)
    
    successful_results = [r for r in all_results if r is not None]
    
    # Create combined outputs in tile folder
    all_features = []
    for result in successful_results:
        if result.get('component_features'):
            all_features.extend(result['component_features'])
    
    if all_features:
        combined_gdf = gpd.GeoDataFrame([
            {'geometry': f['geometry'], **f['properties']} 
            for f in all_features
        ])
        
        # Set CRS from temp file
        if successful_results and successful_results[0]['temp_files']:
            try:
                temp_gdf = gpd.read_file(successful_results[0]['temp_files'][0])
                combined_gdf.crs = temp_gdf.crs
            except:
                pass
        
        # Save combined files in tile folder
        combined_geojson = os.path.join(tile_folder, "combined.geojson")
        combined_csv = os.path.join(tile_folder, "combined.csv")
        
        combined_gdf.to_file(combined_geojson, driver='GeoJSON')
        combined_gdf.drop('geometry', axis=1).to_csv(combined_csv, index=False)
        
        try:
            combined_parquet = os.path.join(tile_folder, "combined.parquet")
            combined_gdf.to_parquet(combined_parquet)
        except:
            pass
    
    # Cleanup temp files
    for result in successful_results:
        for temp_file in result.get('temp_files', []):
            try:
                os.remove(temp_file)
            except:
                pass
    
    total_components = sum(r['n_components'] for r in successful_results)
    total_area = sum(r['total_area_hectares'] for r in successful_results)
    
    print(f"  → {len(successful_results)}/{n_samples} samples, {total_components} components, {total_area:.1f} ha")
    
    return {
        'tile_name': tile_name,
        'successful_samples': len(successful_results),
        'total_components': total_components,
        'total_area_hectares': total_area,
        'component_features': all_features
    }

def process_multiple_rasters(input_folder, min_area_pixels, confidence_level,
                           n_samples, window_size, output_folder, seed, n_processes):
    """Process all TIF files in folder"""
    
    tif_files = get_tif_files(input_folder)
    if not tif_files:
        print(f"No TIF files found in {input_folder}")
        return None
    
    print(f"Found {len(tif_files)} TIF files")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get folder prefix from input folder name
    folder_prefix = Path(input_folder).name
    
    all_results = []
    for i, tif_file in enumerate(tif_files, 1):
        print(f"[{i}/{len(tif_files)}] Processing {Path(tif_file).name}")
        
        result = process_random_samples_parallel(
            input_path=tif_file,
            min_area_pixels=min_area_pixels,
            confidence_level=confidence_level,
            n_samples=n_samples,
            window_size=window_size,
            output_folder=output_folder,
            seed=seed + i,
            n_processes=n_processes
        )
        
        if result:
            all_results.append(result)
    
    # Create final combined parquet with folder prefix
    if all_results:
        all_features = []
        for result in all_results:
            all_features.extend(result.get('component_features', []))
        
        if all_features:
            final_gdf = gpd.GeoDataFrame([
                {'geometry': f['geometry'], **f['properties']} 
                for f in all_features
            ])
            
            # Save final combined files with folder prefix
            final_parquet = os.path.join(output_folder, f"{folder_prefix}_combined.parquet")
            final_csv = os.path.join(output_folder, f"{folder_prefix}_combined.csv")
            final_geojson = os.path.join(output_folder, f"{folder_prefix}_combined.geojson")
            
            try:
                final_gdf.to_parquet(final_parquet)
                print(f"Created final combined parquet: {final_parquet}")
            except Exception as e:
                print(f"Could not create final parquet: {e}")
            
            final_gdf.to_file(final_geojson, driver='GeoJSON')
            final_gdf.drop('geometry', axis=1).to_csv(final_csv, index=False)
            print(f"Created final combined files: {final_csv}, {final_geojson}")
    
    # Summary
    total_samples = sum(r['successful_samples'] for r in all_results)
    total_components = sum(r['total_components'] for r in all_results)
    total_area = sum(r['total_area_hectares'] for r in all_results)
    
    print(f"\nSummary: {len(all_results)} tiles, {total_samples} samples, {total_components} components, {total_area:.1f} ha")
    
    return all_results

# Main execution
if __name__ == "__main__":
    
    # Configuration
    MIN_AREA_PIXELS = 100
    CONFIDENCE_LEVEL = 'high'
    OUTPUT_FOLDER = "data/interim/radd"
    N_SAMPLES = 40
    WINDOW_SIZE = 5000
    SEED = 42
    N_PROCESSES = 40
    
    # Choose mode
    PROCESS_SINGLE_FILE = False  # Set to False for folder mode
    INPUT_RASTER = "data/raw/10N_080W.tif"
    INPUT_FOLDER = "data/raw/radd/south_america"

    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Samples per tile: {N_SAMPLES}")
    print(f"Files per tile: {N_SAMPLES*2} rasters + 3 combined files")
    
    if PROCESS_SINGLE_FILE:
        
        result = process_random_samples_parallel(
            input_path=INPUT_RASTER,
            min_area_pixels=MIN_AREA_PIXELS,
            confidence_level=CONFIDENCE_LEVEL,
            n_samples=N_SAMPLES,
            window_size=WINDOW_SIZE,
            output_folder=OUTPUT_FOLDER,
            seed=SEED,
            n_processes=N_PROCESSES
        )
        
        if result:
            print(f"Complete: {result['successful_samples']} samples processed")
    
    else:
        INPUT_FOLDER = "data/raw/radd/south_america"
        
        results = process_multiple_rasters(
            input_folder=INPUT_FOLDER,
            min_area_pixels=MIN_AREA_PIXELS,
            confidence_level=CONFIDENCE_LEVEL,
            n_samples=N_SAMPLES,
            window_size=WINDOW_SIZE,
            output_folder=OUTPUT_FOLDER,
            seed=SEED,
            n_processes=N_PROCESSES
        )
        
        if results:
            print(f"Complete: {len(results)} tiles processed")