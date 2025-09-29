import os
import sys
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import cubo
import xarray as xr
import rioxarray as rxr
from datetime import datetime, timedelta
import rootutils

root_path = rootutils.find_root()

def load_dynamic_world(lat, lon, start_date, end_date, edge_size=128):
    """Load Dynamic World forest data"""
    da_dw = cubo.create(
        lat=lat, lon=lon,
        collection="GOOGLE/DYNAMICWORLD/V1",
        bands=["trees"],
        start_date=start_date, end_date=end_date,
        edge_size=edge_size, resolution=10, gee=True
    )
    da_dw = da_dw.drop_duplicates(dim='time', keep='first')
    da_dw = da_dw.rio.write_crs(f"EPSG:{da_dw.epsg}")
    return da_dw

def date_to_radd_days(date_str):
    """Convert date string to RADD day encoding"""
    base_date = datetime(2014, 12, 31)
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    return (target_date - base_date).days

def extract_128x128_window(lat, lon, window_size=128, pixel_size_meters=10):
    """Calculate bounds for 128x128 window around centroid"""
    half_extent = (window_size * pixel_size_meters) / 2
    # Convert to degrees (rough approximation)
    deg_extent = half_extent / 111000
    return (lon - deg_extent, lat - deg_extent, lon + deg_extent, lat + deg_extent)

def process_single_event(event_idx, event_row, output_base_dir, window_size=128):
    """Process one deforestation event"""
    
    print(f"Processing event {event_idx}: {event_row['tile_name']}, component {event_row['component_id']}")
    
    event_dir = Path(output_base_dir) / f"event_{event_idx:06d}"
    event_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract event info
        center_lat = event_row['centroid_y']
        center_lon = event_row['centroid_x']
        tile_name = event_row['tile_name']
        component_id = event_row['component_id']
        earliest_alert = event_row['earliest_alert']
        latest_alert = event_row['latest_alert']
        
        # Find component file
        radd_interim_dir = root_path / f"data/interim/radd/{tile_name}"
        component_file = None
        for comp_file in radd_interim_dir.glob("*_components.tif"):
            comp_data = rxr.open_rasterio(comp_file, chunks=True)
            if component_id in np.unique(comp_data.values):
                component_file = comp_file
                comp_data.close()
                break
            comp_data.close()
        
        if component_file is None:
            raise ValueError(f"Component {component_id} not found")
        
        # Get window bounds
        minx, miny, maxx, maxy = extract_128x128_window(center_lat, center_lon, window_size)
        
        # Load component data window
        comp_da = rxr.open_rasterio(component_file)
        comp_window = comp_da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        # Load corresponding RADD data window
        radd_file = radd_interim_dir / component_file.name.replace("_components.tif", "_dates.tif")
        radd_da = rxr.open_rasterio(radd_file)
        radd_window = radd_da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        # Create deforestation mask
        component_mask = comp_window == component_id
        
        # Temporal filtering
        start_days = date_to_radd_days(earliest_alert)
        end_days = date_to_radd_days(latest_alert)
        valid_mask = radd_window > 0
        temporal_mask = valid_mask & (radd_window >= start_days) & (radd_window <= end_days)
        
        deforestation_mask = component_mask & temporal_mask
        
        # Get Dynamic World data
        start_date = (datetime.strptime(earliest_alert, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(latest_alert, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        
        da_dw = load_dynamic_world(center_lat, center_lon, start_date, end_date)
        
        # Extract forest mask (take most recent)
        trees_data = da_dw.isel(time=-1).sel(band='trees')
        
        # Crop to center 128x128
        h, w = trees_data.shape
        center_y, center_x = h // 2, w // 2
        half_win = window_size // 2
        forest_data = trees_data[center_y-half_win:center_y+half_win, center_x-half_win:center_x+half_win]
        forest_mask = forest_data > 0.5
        
        # Reproject deforestation mask to Dynamic World CRS
        target_crs = trees_data.rio.crs
        deforestation_reprojected = deforestation_mask.rio.reproject_match(forest_data, resampling=rasterio.enums.Resampling.nearest)
        
        # Create 3-class segmentation mask
        segmentation_mask = np.zeros((window_size, window_size), dtype=np.uint8)
        segmentation_mask[deforestation_reprojected.values] = 2  # deforestation
        stable_forest = forest_mask.values & ~deforestation_reprojected.values
        segmentation_mask[stable_forest] = 1  # stable forest
        # class 0 (no forest) is already set by np.zeros
        
        # Save segmentation mask
        mask_file = event_dir / "segmentation_mask.tif"
        with rasterio.open(
            mask_file, 'w', driver='GTiff', height=window_size, width=window_size,
            count=1, dtype=segmentation_mask.dtype, crs=target_crs,
            transform=forest_data.rio.transform(), compress='lzw'
        ) as dst:
            dst.write(segmentation_mask, 1)
        
        # Save metadata
        metadata = {
            'event_idx': event_idx,
            'centroid_lat': center_lat,
            'centroid_lon': center_lon,
            'tile_name': tile_name,
            'component_id': int(component_id),
            'earliest_alert': earliest_alert,
            'latest_alert': latest_alert,
            'crs': str(target_crs),
            'class_counts': {
                'no_forest': int(np.sum(segmentation_mask == 0)),
                'stable_forest': int(np.sum(segmentation_mask == 1)),
                'deforestation': int(np.sum(segmentation_mask == 2))
            }
        }
        
        import json
        with open(event_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Success: {metadata['class_counts']}")
        return {'event_idx': event_idx, 'success': True, 'class_counts': metadata['class_counts']}
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {'event_idx': event_idx, 'success': False, 'error': str(e)}

def main():
    """Main processing function"""
    
    print("🏷️  Creating segmentation labels")
    print("=" * 50)
    
    OUTPUT_DIR = root_path / "data/processed/segmentation_labels"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load events
    deforest_df = pd.read_csv(root_path / "data/processed/radd/south_america_combined_clean.csv")
    print(f"✓ Loaded {len(deforest_df)} events")
    
    # Process events
    results = []
    for idx, (_, event_row) in enumerate(deforest_df.head(5).iterrows()):
        result = process_single_event(idx, event_row, OUTPUT_DIR)
        results.append(result)
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n📊 Summary: {len(successful)} success, {len(failed)} failed")
    
    if successful:
        total_classes = {'no_forest': 0, 'stable_forest': 0, 'deforestation': 0}
        for result in successful:
            for class_name, count in result['class_counts'].items():
                total_classes[class_name] += count
        
        print("🎯 Class distribution:")
        for class_name, count in total_classes.items():
            print(f"  {class_name}: {count:,} pixels")

if __name__ == "__main__":
    main()