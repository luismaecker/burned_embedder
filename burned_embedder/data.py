import cubo
import numpy as np
import datetime
import pandas as pd
import datetime
from datetime import timedelta

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
        #print("No post-baseline change data found, returning original data")
        return data
    
    # Find which bands in our data need harmonization
    available_bands = data.band.values.tolist()
    bands_to_process = [b for b in bands_to_harmonize if b in available_bands]
    
    if not bands_to_process:
      #  print("No bands requiring harmonization found")
        return data
    
    # print(f"Applying baseline harmonization to bands: {bands_to_process}")
    
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


def load_s1(lat, lon, start_date="2024-01-01", end_date="2025-12-31", edge_size=128, verbose=False):
    """Load and clean Sentinel-1 data"""

    if verbose:
        print("Loading Sentinel-1 data...")
    da_s1 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        start_date=start_date, end_date=end_date,
        edge_size=edge_size, resolution=10,
       # query={"sat:orbit_state": {"eq": "ascending"}}

    )

    # Clean duplicates
    da_s1_clean = da_s1.drop_duplicates(dim='time', keep='first')
    
    if verbose:
        print(f"S1 data shape: {da_s1_clean.shape}")
    
    return da_s1_clean

def load_s2(lat, lon, start_date="2024-01-01", end_date="2025-12-31", bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
            edge_size=100, max_cloud_cover=10, verbose = False):
    """Load and clean Sentinel-2 data with automatic baseline correction"""

    if verbose:
        print("Loading Sentinel-2 data...")

    da_s2 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-2-l2a",
        bands=bands,
        start_date=start_date, end_date=end_date,
        edge_size=edge_size, resolution=10,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )

    # Clean duplicates
    da_s2_clean = da_s2.drop_duplicates(dim='time', keep='first')
    
    # Apply baseline harmonization
    da_s2_harmonized = harmonize_s2_baseline(da_s2_clean)

    if verbose:
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






def find_closest_observations(timestamps, deforest_start, deforest_end, n_before=1, n_after=1):
    """
    Find n closest observations before deforestation start and after deforestation end.
    
    Args:
        timestamps: Array of observation timestamps
        deforest_start: Start date of deforestation period
        deforest_end: End date of deforestation period  
        n_before: Number of observations to get before deforestation
        n_after: Number of observations to get after deforestation
        
    Returns:
        Tuple of (before_indices, after_indices)
    """
    import numpy as np
    
    # Convert to datetime if needed
    if isinstance(deforest_start, str):
        deforest_start = pd.to_datetime(deforest_start)
    if isinstance(deforest_end, str):
        deforest_end = pd.to_datetime(deforest_end)
    
    # Convert timestamps to pandas datetime if needed
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)
    
    # Find observations before and after deforestation period
    before_mask = timestamps < deforest_start
    after_mask = timestamps > deforest_end
    
    before_times = timestamps[before_mask]
    after_times = timestamps[after_mask]
    
    # Get indices of closest observations
    before_indices = np.array([], dtype=int)
    after_indices = np.array([], dtype=int)
    
    if len(before_times) > 0:
        # Calculate time differences and convert to numpy array
        time_diffs = (deforest_start - before_times).total_seconds()
        time_diffs_array = np.abs(time_diffs.values)  # Convert to numpy and take absolute value
        sorted_before_idx = np.argsort(time_diffs_array)[:n_before]
        before_indices = np.where(before_mask)[0][sorted_before_idx]
    
    if len(after_times) > 0:
        # Calculate time differences and convert to numpy array
        time_diffs = (after_times - deforest_end).total_seconds()
        time_diffs_array = np.abs(time_diffs.values)  # Convert to numpy and take absolute value
        sorted_after_idx = np.argsort(time_diffs_array)[:n_after]
        after_indices = np.where(after_mask)[0][sorted_after_idx]
    
    return before_indices.astype(int), after_indices.astype(int)

def load_s1_filtered(lat, lon, start_date, end_date, deforest_start, deforest_end, 
                    n_before=1, n_after=1, edge_size=100):
    """Load Sentinel-1 data and filter to specific observations around deforestation period"""
    
    # Load all data first
    da_s1_full = load_s1(lat, lon, start_date, end_date, edge_size)
    
    if da_s1_full is None or len(da_s1_full.time) == 0:
        print(f"    Found {len(da_s1_full.time)} total S1 observations")
        print(f"    Warning: No S1 data available in search window")
        return None
    
    
    # Find closest observations
    timestamps = pd.to_datetime(da_s1_full.time.values)
    before_idx, after_idx = find_closest_observations(
        timestamps, deforest_start, deforest_end, n_before, n_after
    )
    
    
    # Combine indices and select those observations
    selected_indices = np.concatenate([before_idx, after_idx])
    if len(selected_indices) > 0:
        da_s1_filtered = da_s1_full.isel(time=selected_indices)
        return da_s1_filtered.sortby('time')
    else:
        print(f"    Warning: No suitable observations found")
        return None
    
    


def calculate_search_dates(earliest_alert, latest_alert, buffer_months=1):
    """
    Calculate search start and end dates based on deforestation alerts with buffer.
    
    Args:
        earliest_alert: First deforestation alert date
        latest_alert: Last deforestation alert date  
        buffer_months: Number of months to add/subtract as buffer
        
    Returns:
        Tuple of (search_start_date, search_end_date) as strings
    """
    # Convert to datetime if strings
    if isinstance(earliest_alert, str):
        earliest_alert = pd.to_datetime(earliest_alert)
    if isinstance(latest_alert, str):
        latest_alert = pd.to_datetime(latest_alert)
    
    # Calculate buffer in days (approximate)
    buffer_days = buffer_months * 30
    
    # Add buffer before and after
    search_start = earliest_alert - timedelta(days=buffer_days)
    search_end = latest_alert + timedelta(days=buffer_days)
    
    # Convert back to string format
    search_start_str = search_start.strftime("%Y-%m-%d")
    search_end_str = search_end.strftime("%Y-%m-%d")
    
    return search_start_str, search_end_str


def clean_metadata_nc(da, metadata_dict=None):
    """Remove problematic coordinates and add metadata to NetCDF"""
    # Predefined sets of coordinates to drop for each satellite type
    S1_COORDS_TO_DROP = {'sar:polarizations', 'raster:bands', 'description', 'title', "proj:centroid", 'proj:shape', 
                         'proj:transform', 's1:shape', "proj:bbox"}
    S2_COORDS_TO_DROP = {'title', 'proj:bbox', 'proj:shape', 'proj:transform', "proj:centroid"}
    
    # Determine which set to use based on what coordinates exist
    if 'sar:polarizations' in da.coords:
        coords_to_drop = [coord for coord in S1_COORDS_TO_DROP if coord in da.coords]
    else:
        coords_to_drop = [coord for coord in S2_COORDS_TO_DROP if coord in da.coords]
    
    # Drop coordinates if any exist
    da_clean = da.drop_vars(coords_to_drop) if coords_to_drop else da
    
    # Add metadata as attributes if provided
    if metadata_dict is not None:
        da_clean.attrs.update(metadata_dict)
    
    return da_clean