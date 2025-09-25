import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm


def validate_timestep(data_slice, threshold=0.5):
    """
    Validate a single timestep
    
    Args:
        data_slice: 3D array (bands, height, width) for one timestep
        threshold: Maximum fraction of invalid pixels allowed
        
    Returns:
        is_valid: bool
        invalid_fraction: float
    """
    total_pixels = data_slice.size
    invalid_pixels = np.sum(np.isnan(data_slice)) + np.sum(np.isinf(data_slice))
    invalid_fraction = invalid_pixels / total_pixels
    
    return invalid_fraction <= threshold, invalid_fraction


def find_invalid_timesteps(data_array, threshold=0.5, use_dask=True):
    """
    Find timesteps with too many invalid pixels
    
    Args:
        data_array: xarray DataArray with time dimension
        threshold: Maximum fraction of invalid pixels allowed
        use_dask: Whether to use Dask for parallel processing
        
    Returns:
        invalid_indices: List of timestep indices to drop
        invalid_fractions: List of invalid fractions for dropped timesteps
    """
    print(f"Checking {len(data_array.time)} timesteps for invalid data...")
    
    invalid_indices = []
    invalid_fractions = []
    
    if use_dask and hasattr(data_array.data, 'chunks'):
        print("Using Dask for parallel validation...")
        # Process in parallel with Dask
        for t in tqdm(range(len(data_array.time))):
            timestep_data = data_array.isel(time=t).values
            is_valid, invalid_frac = validate_timestep(timestep_data, threshold)
            
            if not is_valid:
                invalid_indices.append(t)
                invalid_fractions.append(invalid_frac)
    else:
        # Sequential processing
        for t in tqdm(range(len(data_array.time))):
            timestep_data = data_array.isel(time=t).values
            is_valid, invalid_frac = validate_timestep(timestep_data, threshold)
            
            if not is_valid:
                invalid_indices.append(t)
                invalid_fractions.append(invalid_frac)
    
    print(f"Found {len(invalid_indices)} invalid timesteps")
    return invalid_indices, invalid_fractions


def clean_data_array(data_array, invalid_threshold=0.5, fill_value=0.0):
    """
    Clean a data array by removing bad timesteps and filling invalid values
    
    Args:
        data_array: xarray DataArray  
        invalid_threshold: Drop timesteps with more than this fraction invalid
        fill_value: Value to replace NaN/inf with
        
    Returns:
        cleaned_array: Clean DataArray
        dropped_indices: List of dropped timestep indices
    """
    print(f"Cleaning data array: {data_array.shape}")
    
    # Find invalid timesteps
    invalid_indices, invalid_fractions = find_invalid_timesteps(
        data_array, invalid_threshold)
    
    # Log dropped timesteps
    for idx, frac in zip(invalid_indices, invalid_fractions):
        timestamp = data_array.time.values[idx]
        print(f"Dropping timestep {idx} ({timestamp}): {frac:.2%} invalid pixels")
    
    # Keep only valid timesteps
    valid_indices = [i for i in range(len(data_array.time)) if i not in invalid_indices]
    clean_array = data_array.isel(time=valid_indices)
    
    # Replace remaining NaN/inf values
    clean_values = np.nan_to_num(clean_array.values, 
                                nan=fill_value, 
                                posinf=fill_value, 
                                neginf=fill_value)
    clean_array = clean_array.copy(data=clean_values)
    
    print(f"Cleaned array shape: {clean_array.shape}")
    print(f"Dropped {len(invalid_indices)} timesteps")
    
    return clean_array, invalid_indices


def find_timestamp_pairs(s1_times, s2_times, max_diff_days=2):
    """Find matching timestamp pairs"""
    print(f"Finding timestamp pairs within {max_diff_days} days...")
    
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
    
    print(f"Found {len(final_pairs)} timestamp pairs")
    return final_pairs


def preprocess_satellite_data(da_s1, da_s2, invalid_threshold=0.5, 
                             max_diff_days=2, fill_value=0.0):
    """
    Complete preprocessing pipeline
    
    Args:
        da_s1: Raw S1 DataArray
        da_s2: Raw S2 DataArray
        invalid_threshold: Drop timesteps with more invalid pixels than this
        max_diff_days: Maximum days between S1/S2 acquisitions
        fill_value: Value to replace invalid pixels with
        
    Returns:
        da_s1_clean: Cleaned S1 data
        da_s2_clean: Cleaned S2 data
        timestamp_pairs: Valid (s2_idx, s1_idx) pairs
        stats: Preprocessing statistics
    """
    print("="*60)
    print("PREPROCESSING SATELLITE DATA")
    print("="*60)
    
    original_s1_timesteps = len(da_s1.time)
    original_s2_timesteps = len(da_s2.time)
    
    # Clean S1 data
    da_s1_clean, dropped_s1 = clean_data_array(da_s1, invalid_threshold, fill_value)
    
    # Clean S2 data
    da_s2_clean, dropped_s2 = clean_data_array(da_s2, invalid_threshold, fill_value)
    
    # Find valid timestamp pairs
    timestamp_pairs = find_timestamp_pairs(
        da_s1_clean.time.values, da_s2_clean.time.values, max_diff_days)
    
    # Statistics
    stats = {
        'original_s1_timesteps': original_s1_timesteps,
        'original_s2_timesteps': original_s2_timesteps,
        'clean_s1_timesteps': len(da_s1_clean.time),
        'clean_s2_timesteps': len(da_s2_clean.time),
        'dropped_s1_timesteps': len(dropped_s1),
        'dropped_s2_timesteps': len(dropped_s2),
        'timestamp_pairs': len(timestamp_pairs),
        'data_retention_s1': len(da_s1_clean.time) / original_s1_timesteps,
        'data_retention_s2': len(da_s2_clean.time) / original_s2_timesteps,
    }
    
    print("="*60)
    print("PREPROCESSING SUMMARY:")
    for key, value in stats.items():
        if 'retention' in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    print("="*60)
    
    return da_s1_clean, da_s2_clean, timestamp_pairs, stats