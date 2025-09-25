import cubo
import numpy as np
import datetime

def harmonize_s2_baseline(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.
    
    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with Sentinel-2 data
        
    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    
    # Bands that need harmonization
    bands_to_harmonize = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", 
        "B08", "B8A", "B09", "B10", "B11", "B12"
    ]
    
    # Check if we have any data after the cutoff date
    post_cutoff_mask = data.time >= np.datetime64(cutoff)
    
    if not post_cutoff_mask.any():
        print("No post-baseline change data found, returning original data")
        return data
    
    # Find which bands in our data need harmonization
    available_bands = data.band.values.tolist()
    bands_to_process = [b for b in bands_to_harmonize if b in available_bands]
    
    if not bands_to_process:
        print("No bands requiring harmonization found")
        return data
    
    print(f"Applying baseline harmonization to bands: {bands_to_process}")
    
    # Create a copy to avoid modifying original data
    harmonized = data.copy()
    
    # Apply harmonization to post-cutoff data for specified bands
    for band in bands_to_process:
        band_data = harmonized.sel(band=band)
        post_cutoff_data = band_data.where(post_cutoff_mask)
        
        # Apply offset correction: clip to offset value, then subtract offset
        corrected = post_cutoff_data.clip(min=offset) - offset
        
        # Update the harmonized array
        harmonized.loc[dict(band=band)] = band_data.where(~post_cutoff_mask, corrected)
    
    return harmonized

def load_s1(lat, lon, start_date="2024-01-01", end_date="2025-12-31"):
    """Load and clean Sentinel-1 data"""

    print("Loading Sentinel-1 data...")
    da_s1 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        start_date=start_date, end_date=end_date,
        edge_size=128, resolution=10,
       # query={"sat:orbit_state": {"eq": "ascending"}}

    )

    # Clean duplicates
    da_s1_clean = da_s1.drop_duplicates(dim='time', keep='first')
    
    print(f"S1 data shape: {da_s1_clean.shape}")
    
    return da_s1_clean

def load_s2(lat, lon, start_date="2024-01-01", end_date="2025-12-31"):
    """Load and clean Sentinel-2 data with automatic baseline correction"""
    print("Loading Sentinel-2 data...")
    da_s2 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-2-l2a",
        bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
        start_date=start_date, end_date=end_date,
        edge_size=128, resolution=10,
        query={"eo:cloud_cover": {"lt": 2}}
    )

    # Clean duplicates
    da_s2_clean = da_s2.drop_duplicates(dim='time', keep='first')
    
    # Apply baseline harmonization
    da_s2_harmonized = harmonize_s2_baseline(da_s2_clean)

    print(f"S2 data shape: {da_s2_harmonized.shape}")

    return da_s2_harmonized


def find_closest_timestamps(s1_times, s2_times, max_diff_days=2):
    """Find S1 and S2 acquisitions within max_diff_days of each other"""
    pairs = []
    s1_times_dt = [np.datetime64(t).astype('datetime64[D]') for t in s1_times]
    s2_times_dt = [np.datetime64(t).astype('datetime64[D]') for t in s2_times]
    
    for i, s2_time in enumerate(s2_times_dt):
        for j, s1_time in enumerate(s1_times_dt):
            diff_days = abs((s2_time - s1_time).astype(int))
            if diff_days <= max_diff_days:
                pairs.append((i, j, diff_days))
    
    # Sort by difference and take best matches
    pairs.sort(key=lambda x: x[2])
    used_s1, used_s2 = set(), set()
    final_pairs = []
    
    for s2_idx, s1_idx, diff in pairs:
        if s2_idx not in used_s2 and s1_idx not in used_s1:
            final_pairs.append((s2_idx, s1_idx))
            used_s2.add(s2_idx)
            used_s1.add(s1_idx)
    
    return final_pairs