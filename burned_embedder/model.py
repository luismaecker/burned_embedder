import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from burned_embedder.analysis import calculate_ndvi


def load_model(device, model_size='base', weights_path=None):
    """Load and prepare the TerraFM model"""
    try:
        from terrafm import terrafm_base, terrafm_large
    except ImportError:
        raise ImportError("TerraFM module not found. Please ensure terrafm.py is in your Python path.")
    
    print(f"Loading TerraFM-{model_size.upper()} model...")
    
    # Load the appropriate model
    if model_size == 'base':
        encoder = terrafm_base()
    elif model_size == 'large':
        encoder = terrafm_large()
    else:
        raise ValueError("model_size must be 'base' or 'large'")
    
    # Load pretrained weights if provided
    if weights_path:
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights: {msg}")
    
    encoder.to(device)
    encoder.eval()
    return encoder


def process_batch_unified(encoder, da_s1_batch, da_s2_batch, acquisition_dates_batch, mode, device, lon, lat):
    """Unified batch processing function for TerraFM"""
    
    # TerraFM expects separate inputs for different modalities
    if mode == 's1_only':
        # Only S1 data (2 channels: VV, VH)
        s1_data_list = [np.array(da_s1_t) for da_s1_t in da_s1_batch]
        s1_batch = np.stack(s1_data_list, axis=0)  # Shape: (batch_size, 2, H, W)
        x = torch.from_numpy(s1_batch).float().to(device)
        
    elif mode == 's2_only':
        # Only S2 data (12 channels based on your data shape)
        s2_data_list = [da_s2_t.values for da_s2_t in da_s2_batch]
        s2_batch = np.stack(s2_data_list, axis=0)  # Shape: (batch_size, 12, H, W)
        x = torch.from_numpy(s2_batch).float().to(device)
        
    elif mode == 'combined':
        # For combined mode, you need to check TerraFM's expected input format
        # It might need both S1 and S2 as separate arguments
        s1_data_list = [np.array(da_s1_t) for da_s1_t in da_s1_batch]
        s2_data_list = [da_s2_t.values for da_s2_t in da_s2_batch]
        
        s1_batch = np.stack(s1_data_list, axis=0)  # Shape: (batch_size, 2, H, W)
        s2_batch = np.stack(s2_data_list, axis=0)  # Shape: (batch_size, 12, H, W)
        
        # Concatenate S1 and S2 (14 channels total: 2 S1 + 12 S2)
        data_batch = np.concatenate([s1_batch, s2_batch], axis=1)
        x = torch.from_numpy(data_batch).float().to(device)
    
    else:
        raise ValueError("Mode must be 's1_only', 's2_only', or 'combined'")
    
    # TerraFM forward pass
    with torch.no_grad():
        embeddings = encoder(x)
    
    return embeddings.cpu().numpy()

def process_embeddings_batched(encoder, da_s1, da_s2, timestamp_pairs, mode='combined', batch_size=None, device=None, lon=None, lat=None):
    """Process embeddings in batches and filter out NaN results"""
    
    # Set default batch sizes based on mode
    if batch_size is None:
        default_batch_sizes = {'s1_only': 64, 's2_only': 32, 'combined': 16}
        batch_size = default_batch_sizes[mode]
    
    # For s1_only mode, use ALL S1 timesteps instead of just matched pairs
    if mode == 's1_only':
        num_timesteps = len(da_s1.time)
        print(f"Processing {num_timesteps} timesteps for {mode} mode with batch size {batch_size}...")
        
        embeddings_list = []
        ndvi_list = []
        dates_list = []
        
        # Process all S1 timesteps in batches
        for i in tqdm(range(0, num_timesteps, batch_size)):
            batch_end = min(i + batch_size, num_timesteps)
            batch_indices = list(range(i, batch_end))
            
            # Prepare batch data using S1 indices only
            da_s1_batch = [da_s1.isel(time=idx) for idx in batch_indices]
            da_s2_batch = [None] * len(batch_indices)  # Not used for s1_only
            acquisition_dates_batch = [pd.to_datetime(da_s1.time.values[idx]).to_pydatetime() 
                                     for idx in batch_indices]
            
            try:
                # Process batch
                batch_embeddings = process_batch_unified(encoder, da_s1_batch, da_s2_batch, 
                                                       acquisition_dates_batch, mode, device, 
                                                       lon, lat)
                
                # For s1_only mode, we can't calculate NDVI (no S2 data), so use dummy values
                batch_ndvi = [np.nan] * len(batch_indices)
                batch_dates = [da_s1.time.values[idx] for idx in batch_indices]
                
                # Collect results
                embeddings_list.extend(batch_embeddings)
                ndvi_list.extend(batch_ndvi)
                dates_list.extend(batch_dates)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM with batch size {len(batch_indices)}. Try reducing batch_size parameter.")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
    
    else:
        # Original logic for s2_only and combined modes
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