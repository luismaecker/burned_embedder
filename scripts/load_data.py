from datetime import datetime
import sys
import rootutils
import pandas as pd
from burned_embedder.data import load_s1, load_s2

root_path = rootutils.find_root()
start_date = "2025-01-01"
end_date = "2025-12-31"

def clean_for_netcdf(da):
    """Remove problematic coordinates that can't be serialized to NetCDF"""
    # List of coordinates that typically cause serialization issues
    problematic_coords = [
        'sar:polarizations', 
        'raster:bands',
        'proj:centroid',
        'description',
        'title',
        "proj:bbox"
    ]
    
    # Drop problematic coordinates
    coords_to_drop = [coord for coord in problematic_coords if coord in da.coords]
    
    if coords_to_drop:
        print(f"    Dropping problematic coordinates: {coords_to_drop}")
        da_clean = da.drop_vars(coords_to_drop)
    else:
        da_clean = da
    
    return da_clean

def main():
    """Download satellite data for all fires"""
    print("Starting Fire Data Download")
    print("=" * 50)

    # Load fire data
    fire_df = pd.read_parquet("data/raw/spain_fires_25.parquet")
    print(f"Loaded {len(fire_df)} fire records")
    
    # Process each fire in the CSV
    for csv_row_idx, row in fire_df.iterrows():
        print(f"\nProcessing fire {csv_row_idx + 1}/{len(fire_df)} (row index: {csv_row_idx})")
        
        # Extract coordinates and time from the CSV row
        LAT = row['Latitude']
        LON = row['Longitude']
        fire_time = row['Time']
        
        print(f"  Coordinates: LAT={LAT:.6f}, LON={LON:.6f}")
        print(f"  Fire time: {fire_time}")
        
        # Create directory for this specific fire
        fire_data_dir = root_path / "data" / "raw" / str(csv_row_idx)
        fire_data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load Sentinel-1 data
            print(f"  Loading Sentinel-1 data...")
            da_s1 = load_s1(LAT, LON, start_date=start_date, end_date=end_date)
            da_s1_clean = clean_for_netcdf(da_s1)
            s1_file = fire_data_dir / "da_s1.nc"
            da_s1_clean.to_netcdf(s1_file)
            print(f"  ✓ S1 data saved: {s1_file}")
            
            # Load Sentinel-2 data
            # print(f"  Loading Sentinel-2 data...")
            # da_s2 = load_s2(LAT, LON, start_date=start_date, end_date=end_date)
            # da_s2_clean = clean_for_netcdf(da_s2)
            # s2_file = fire_data_dir / "da_s2.nc"
            # da_s2_clean.to_netcdf(s2_file)
            # print(f"  ✓ S2 data saved: {s2_file}")
            
            #print(f"  Data shapes - S1: {da_s1.shape}, S2: {da_s2.shape}")
            
        except Exception as e:
            print(f"  ✗ Error processing fire {csv_row_idx}: {e}")
            continue
    
    print(f"\nData download complete!")

if __name__ == "__main__":
    main()