from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base
from tqdm import tqdm

from burned_embedder.analysis import calculate_ndvi


def load_model(device):
    """Load and prepare the Copernicus FM model"""
    print("Loading Copernicus FM model...")
    encoder = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
    encoder.to(device)
    encoder.eval()
    return encoder


def process_s1_only(encoder, da_s1_t, acquisition_date, device, lon, lat, area, kernel_size, s1_wavelengths, s1_bandwidths):
    """Process S1-only data through the model"""
    s1_data = np.array(da_s1_t)  # Shape: (2, H, W)
    x = torch.from_numpy(s1_data).float().unsqueeze(0).to(device)
    
    # Calculate time delta in days since 1970-01-01
    reference_date = datetime(1970, 1, 1)
    time_delta_days = (acquisition_date - reference_date).days
    
    # Create metadata tensor: [lon, lat, time_delta_days, area_km2]
    metadata = torch.tensor([[lon, lat, float(time_delta_days), area]], 
                           dtype=torch.float32, device=device)
    
    with torch.no_grad():
        embedding = encoder(x, metadata, wavelengths=s1_wavelengths, 
                          bandwidths=s1_bandwidths, language_embed=None, 
                          input_mode='spectral', kernel_size=kernel_size)
    
    return embedding.cpu().numpy().flatten()


def process_s2_only(encoder, da_s2_t, acquisition_date, device, lon, lat, area, kernel_size, s2_wavelengths, s2_bandwidths):
    """Process S2-only data through the model"""
    s2_data = da_s2_t.values  # Shape: (13, H, W) - make sure you have all 13 bands
    x = torch.from_numpy(s2_data).float().unsqueeze(0).to(device)
    
    # Calculate time delta in days since 1970-01-01
    reference_date = datetime(1970, 1, 1)
    time_delta_days = (acquisition_date - reference_date).days
    
    # Create metadata tensor: [lon, lat, time_delta_days, area_km2]
    metadata = torch.tensor([[lon, lat, float(time_delta_days), area]], 
                           dtype=torch.float32, device=device)
    
    with torch.no_grad():
        embedding = encoder(x, metadata, wavelengths=s2_wavelengths, 
                          bandwidths=s2_bandwidths, language_embed=None, 
                          input_mode='spectral', kernel_size=kernel_size)
    
    return embedding.cpu().numpy().flatten()


def process_combined(encoder, da_s1_t, da_s2_t, acquisition_date, device, lon, lat, area, kernel_size, s1_wavelengths, s1_bandwidths, s2_wavelengths, s2_bandwidths):
    """Process combined S1+S2 data through the model"""
    s1_data = np.array(da_s1_t)  # Shape: (2, H, W)
    s2_data = da_s2_t.values     # Shape: (13, H, W) - make sure you have all 13 bands
    
    # Concatenate along channel dimension
    combined_data = np.concatenate([s1_data, s2_data], axis=0)  # Shape: (15, H, W)
    x = torch.from_numpy(combined_data).float().unsqueeze(0).to(device)
    
    # Calculate time delta in days since 1970-01-01
    reference_date = datetime(1970, 1, 1)
    time_delta_days = (acquisition_date - reference_date).days
    
    # Create metadata tensor: [lon, lat, time_delta_days, area_km2]
    metadata = torch.tensor([[lon, lat, float(time_delta_days), area]], 
                           dtype=torch.float32, device=device)
    
    # Combined wavelengths and bandwidths for 15 bands (2 S1 + 13 S2)
    combined_wavelengths = s1_wavelengths + s2_wavelengths
    combined_bandwidths = s1_bandwidths + s2_bandwidths
    
    with torch.no_grad():
        embedding = encoder(x, metadata, wavelengths=combined_wavelengths, 
                          bandwidths=combined_bandwidths, language_embed=None, 
                          input_mode='spectral', kernel_size=kernel_size)
    
    return embedding.cpu().numpy().flatten()


def process_embeddings(encoder, da_s1, da_s2, timestamp_pairs, mode='combined', device=None, lon=None, lat=None, area=None, kernel_size=None, **spectral_params):
    """Process embeddings for specified mode"""
    print(f"Processing {len(timestamp_pairs)} timesteps for {mode} mode...")
    
    embeddings_list = []
    ndvi_list = []
    dates_list = []
    
    for s2_idx, s1_idx in tqdm(timestamp_pairs):
        da_s2_t = da_s2.isel(time=s2_idx)
        da_s1_t = da_s1.isel(time=s1_idx)
        acquisition_date = pd.to_datetime(da_s2.time.values[s2_idx]).to_pydatetime()
        
        # Get embedding based on mode
        if mode == 's1_only':
            embedding = process_s1_only(encoder, da_s1_t, acquisition_date, device, lon, lat, area, kernel_size, 
                                      spectral_params['s1_wavelengths'], spectral_params['s1_bandwidths'])
        elif mode == 's2_only':
            embedding = process_s2_only(encoder, da_s2_t, acquisition_date, device, lon, lat, area, kernel_size,
                                      spectral_params['s2_wavelengths'], spectral_params['s2_bandwidths'])
        elif mode == 'combined':
            embedding = process_combined(encoder, da_s1_t, da_s2_t, acquisition_date, device, lon, lat, area, kernel_size,
                                       spectral_params['s1_wavelengths'], spectral_params['s1_bandwidths'],
                                       spectral_params['s2_wavelengths'], spectral_params['s2_bandwidths'])
        else:
            raise ValueError("Mode must be 's1_only', 's2_only', or 'combined'")
        
        embeddings_list.append(embedding)
        ndvi_list.append(calculate_ndvi(da_s2_t))
        dates_list.append(da_s2.time.values[s2_idx])
    
    return np.vstack(embeddings_list), np.array(ndvi_list), np.array(dates_list)