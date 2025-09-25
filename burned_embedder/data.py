import cubo
import numpy as np


def load_s1(lat, lon, start_date="2022-01-25", end_date="2025-12-31"):
    """Load and clean Sentinel-1 data"""

    print("Loading Sentinel-1 data...")
    da_s1 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        start_date=start_date, end_date=end_date,
        edge_size=128, resolution=10
    )

    # Clean duplicates
    da_s1_clean = da_s1.drop_duplicates(dim='time', keep='first')
    
    print(f"S1 data shape: {da_s1_clean.shape}")
    
    return da_s1_clean

def load_s2(lat, lon, start_date="2022-01-25", end_date="2025-12-31"):
    """Load and clean Sentinel-2 data"""
    print("Loading Sentinel-2 data...")
    da_s2 = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-2-l2a",
        bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
        start_date=start_date, end_date=end_date,
        edge_size=128, resolution=10
    )

    # Clean duplicates
    da_s2_clean = da_s2.drop_duplicates(dim='time', keep='first')

    print(f"S2 data shape: {da_s2_clean.shape}")

    return da_s2_clean


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