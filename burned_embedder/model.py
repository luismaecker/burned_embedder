from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base
from tqdm import tqdm

from burned_embedder import config
from burned_embedder.analysis import calculate_ndvi


def load_model(device):
    """Load and prepare the Copernicus FM model"""
    print("Loading Copernicus FM model...")
    encoder = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
    encoder.to(device)
    encoder.eval()
    return encoder


def process_batch_unified(encoder, da_s1_batch, da_s2_batch, acquisition_dates_batch, mode, device, lon, lat):
    """Unified batch processing function that handles all modes"""
    
    # Prepare data based on mode
    if mode == 's1_only':
        # Only use S1 data
        s1_data_list = [np.array(da_s1_t) for da_s1_t in da_s1_batch]
        data_batch = np.stack(s1_data_list, axis=0)  # Shape: (batch_size, 2, H, W)
        wavelengths = config.S1_WAVELENGTHS
        bandwidths = config.S1_BANDWIDTHS
        
    elif mode == 's2_only':
        # Only use S2 data
        s2_data_list = [da_s2_t.values for da_s2_t in da_s2_batch]
        data_batch = np.stack(s2_data_list, axis=0)  # Shape: (batch_size, 13, H, W)
        wavelengths = config.S2_WAVELENGTHS
        bandwidths = config.S2_BANDWIDTHS
        
    elif mode == 'combined':
        # Combine S1 and S2 data
        s1_data_list = [np.array(da_s1_t) for da_s1_t in da_s1_batch]
        s2_data_list = [da_s2_t.values for da_s2_t in da_s2_batch]
        
        s1_batch = np.stack(s1_data_list, axis=0)  # Shape: (batch_size, 2, H, W)
        s2_batch = np.stack(s2_data_list, axis=0)  # Shape: (batch_size, 13, H, W)
        
        data_batch = np.concatenate([s1_batch, s2_batch], axis=1)  # Shape: (batch_size, 15, H, W)
        wavelengths = config.S1_WAVELENGTHS + config.S2_WAVELENGTHS
        bandwidths = config.S1_BANDWIDTHS + config.S2_BANDWIDTHS
        
    else:
        raise ValueError("Mode must be 's1_only', 's2_only', or 'combined'")
    
    # Convert to tensor
    x = torch.from_numpy(data_batch).float().to(device)
    
    # Calculate time deltas for all dates
    reference_date = datetime(1970, 1, 1)
    time_deltas = [(date - reference_date).days for date in acquisition_dates_batch]
    
    # Create batch metadata tensor
    metadata_list = [[lon, lat, float(time_delta), config.AREA] for time_delta in time_deltas]
    metadata = torch.tensor(metadata_list, dtype=torch.float32, device=device)
    
    # Forward pass
    with torch.no_grad():
        embeddings = encoder(x, metadata, wavelengths=wavelengths, 
                           bandwidths=bandwidths, language_embed=None, 
                           input_mode='spectral', kernel_size=config.KERNEL_SIZE)
    
    return embeddings.cpu().numpy()

def process_embeddings_batched(encoder, da_s1, da_s2, timestamp_pairs, mode='combined', batch_size=None, device=None, lon=None, lat=None):
    """Process embeddings in batches and filter out NaN results"""
    
    # Set default batch sizes based on mode
    if batch_size is None:
        default_batch_sizes = {'s1_only': 64, 's2_only': 32, 'combined': 16}
        batch_size = default_batch_sizes[mode]
    
    print(f"Processing {len(timestamp_pairs)} timesteps for {mode} mode with batch size {batch_size}...")
    
    embeddings_list = []
    ndvi_list = []
    dates_list = []
    
    # Process in batches
    for i in tqdm(range(0, len(timestamp_pairs), batch_size)):
        batch_pairs = timestamp_pairs[i:i+batch_size]
        
        # Prepare batch data
        da_s2_batch = [da_s2.isel(time=s2_idx) for s2_idx, s1_idx in batch_pairs]
        da_s1_batch = [da_s1.isel(time=s1_idx) for s2_idx, s1_idx in batch_pairs]
        acquisition_dates_batch = [pd.to_datetime(da_s2.time.values[s2_idx]).to_pydatetime() 
                                 for s2_idx, s1_idx in batch_pairs]
        
        try:
            # Process batch
            batch_embeddings = process_batch_unified(encoder, da_s1_batch, da_s2_batch, 
                                                   acquisition_dates_batch, mode, device, 
                                                   lon, lat)
            
            # Calculate NDVI and dates for batch
            batch_ndvi = [calculate_ndvi(da_s2_t) for da_s2_t in da_s2_batch]
            batch_dates = [da_s2.time.values[s2_idx] for s2_idx, s1_idx in batch_pairs]
            
            # Collect results (including potential NaN embeddings)
            embeddings_list.extend(batch_embeddings)
            ndvi_list.extend(batch_ndvi)
            dates_list.extend(batch_dates)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM with batch size {len(batch_pairs)}. Try reducing batch_size parameter.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
    
    # Convert to arrays
    embeddings_array = np.vstack(embeddings_list)
    ndvi_array = np.array(ndvi_list)
    dates_array = np.array(dates_list)
    
    # FILTER OUT NAN EMBEDDINGS
    valid_mask = ~np.isnan(embeddings_array).any(axis=1)
    num_invalid = np.sum(~valid_mask)
    
    if num_invalid > 0:
        print(f"Removing {num_invalid} embeddings with NaN values ({num_invalid/len(embeddings_array):.1%})")
        embeddings_clean = embeddings_array[valid_mask]
        ndvi_clean = ndvi_array[valid_mask]  
        dates_clean = dates_array[valid_mask]
    else:
        print("No NaN embeddings found")
        embeddings_clean = embeddings_array
        ndvi_clean = ndvi_array
        dates_clean = dates_array
    
    print(f"Final dataset: {len(embeddings_clean)} valid samples")
    
    return embeddings_clean, ndvi_clean, dates_clean