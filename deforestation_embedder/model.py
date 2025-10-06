from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base
from tqdm import tqdm

from deforestation_embedder import config


def load_copFM(device):
    """Load and prepare the Copernicus FM model"""
    print("Loading Copernicus FM model...")
    encoder = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
    encoder.to(device)
    encoder.eval()
    return encoder


def process_batch_unified(encoder, da_s1_batch, acquisition_dates_batch, device, lon, lat):
    """Process S1 batch through the model"""
    
    s1_data_list = [np.array(da_s1_t) for da_s1_t in da_s1_batch]
    data_batch = np.stack(s1_data_list, axis=0)
    wavelengths = config.S1_WAVELENGTHS
    bandwidths = config.S1_BANDWIDTHS
    
    x = torch.from_numpy(data_batch).float().to(device)
    
    reference_date = datetime(1970, 1, 1)
    time_deltas = [(date - reference_date).days for date in acquisition_dates_batch]
    
    metadata_list = [[lon, lat, torch.nan, config.AREA] for time_delta in time_deltas]
    metadata = torch.tensor(metadata_list, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        embeddings = encoder(x, metadata, wavelengths=wavelengths, 
                           bandwidths=bandwidths, language_embed=None, 
                           input_mode='spectral', kernel_size=config.KERNEL_SIZE)
    
    return embeddings.cpu().numpy()


def process_embeddings_batched(encoder, da_s1, batch_size=64, device=None, lon=None, lat=None):
    """Process embeddings in batches and filter out NaN results"""
    
    num_timesteps = len(da_s1.time)
    print(f"Processing {num_timesteps} timesteps with batch size {batch_size}...")
    
    embeddings_list = []
    dates_list = []
    
    for i in tqdm(range(0, num_timesteps, batch_size)):
        batch_end = min(i + batch_size, num_timesteps)
        batch_indices = list(range(i, batch_end))
        
        da_s1_batch = [da_s1.isel(time=idx) for idx in batch_indices]
        acquisition_dates_batch = [pd.to_datetime(da_s1.time.values[idx]).to_pydatetime() 
                                 for idx in batch_indices]
        
        try:
            batch_embeddings = process_batch_unified(encoder, da_s1_batch, 
                                                   acquisition_dates_batch, device, 
                                                   lon, lat)
            
            batch_dates = [da_s1.time.values[idx] for idx in batch_indices]
            
            embeddings_list.extend(batch_embeddings)
            dates_list.extend(batch_dates)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM with batch size {len(batch_indices)}. Try reducing batch_size parameter.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
    
    embeddings_array = np.vstack(embeddings_list)
    dates_array = np.array(dates_list)
    
    valid_mask = ~np.isnan(embeddings_array).any(axis=1)
    num_invalid = np.sum(~valid_mask)
    
    if num_invalid > 0:
        print(f"Removing {num_invalid} embeddings with NaN values ({num_invalid/len(embeddings_array):.1%})")
        embeddings_clean = embeddings_array[valid_mask]
        dates_clean = dates_array[valid_mask]
    else:
        print("No NaN embeddings found")
        embeddings_clean = embeddings_array
        dates_clean = dates_array
    
    print(f"Final dataset: {len(embeddings_clean)} valid samples")
    
    return embeddings_clean, dates_clean