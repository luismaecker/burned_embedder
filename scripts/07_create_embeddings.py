import numpy as np
import rootutils
import xarray as xr
from tqdm import tqdm

from burned_embedder import utils
from burned_embedder.model import load_model, process_batch_unified

root_path = rootutils.find_root()

def process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
    """Process embeddings for a single event with 2 timestamps using batch processing"""
    
    # Extract metadata from the dataset
    centroid_lat = float(da_s1.attrs.get('centroid_lat', 0))
    centroid_lon = float(da_s1.attrs.get('centroid_lon', 0))
    
    # Handle different numbers of timesteps
    num_timesteps = len(da_s1.time)
    if num_timesteps != 2:
        print(f"\n✗ Event {event_id}: Expected 2 timesteps, got {num_timesteps}. Skipping.")
        return False
    
    timestamps = da_s1.time.values
    
    try:
        import pandas as pd
        
        # Prepare batch with both timesteps
        da_s1_batch = [da_s1.isel(time=0), da_s1.isel(time=1)]
        da_s2_batch = [None, None]
        
        # Convert timestamps to datetime
        acquisition_dates_batch = [
            pd.to_datetime(timestamps[0]).to_pydatetime(),
            pd.to_datetime(timestamps[1]).to_pydatetime()
        ]
        
        # Process both embeddings in a single batch call
        embeddings = process_batch_unified(
            encoder, da_s1_batch, da_s2_batch, acquisition_dates_batch,
            mode='s1_only', device=device, 
            lon=centroid_lon, lat=centroid_lat
        )
        
        # Save embeddings separately
        embedd_before = embeddings[0]
        embedd_after = embeddings[1]
        
        np.save(output_dir / "embedd_before.npy", embedd_before)
        np.save(output_dir / "embedd_after.npy", embedd_after)
        
        # Save metadata
        metadata = {
            'event_id': event_id,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'num_timesteps': num_timesteps,
            'timestamp_before': str(timestamps[0]),
            'timestamp_after': str(timestamps[1]),
            'earliest_alert': da_s1.attrs.get('earliest_alert', ''),
            'latest_alert': da_s1.attrs.get('latest_alert', ''),
            'duration_days': da_s1.attrs.get('duration_days', ''),
            'sample_type': da_s1.attrs.get('sample_type', ''),
            'model_mode': 's1_only'
        }
        
        # Only add size_pixels if it exists (positive samples)
        if 'size_pixels' in da_s1.attrs:
            metadata['size_pixels'] = da_s1.attrs.get('size_pixels', '')
        
        np.save(output_dir / "metadata.npy", metadata)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing event {event_id}: {e}")
        return False

def process_sample_type(sample_type, encoder, device):
    """Process all events for a given sample type (positive or negative)"""
    print(f"\n{'='*70}")
    print(f"Processing {sample_type.upper()} samples")
    print(f"{'='*70}")
    
    # Find all event directories for this sample type
    sen_data_dir = root_path / "data" / "processed" / "sen_data" / sample_type
    
    if not sen_data_dir.exists():
        print(f"✗ Directory not found: {sen_data_dir}")
        return 0, 0
    
    event_dirs = sorted([d for d in sen_data_dir.iterdir() if d.is_dir() and d.name.startswith('event_')])
    
    print(f"Found {len(event_dirs)} {sample_type} events")
    
    # Create output directory
    embeddings_dir = root_path / "data" / "processed" / "embeddings" / sample_type
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    successful_events = 0
    failed_events = 0
    
    # Process each event
    for event_dir in tqdm(event_dirs, desc=f"Processing {sample_type} samples"):
        event_name = event_dir.name
        event_id = int(event_name.split('_')[1])
        
        # Check if S1 data exists
        s1_file = event_dir / "da_s1.nc"
        if not s1_file.exists():
            print(f"\n✗ S1 file not found for {event_name}")
            failed_events += 1
            continue
        
        # Create output directory for this event
        output_dir = embeddings_dir / event_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if embeddings already exist
        if (output_dir / "embedd_before.npy").exists() and (output_dir / "embedd_after.npy").exists():
            successful_events += 1
            continue
        
        try:
            # Load S1 data
            da_s1 = xr.open_dataset(s1_file)['sentinel-1-rtc']
            
            # Process embeddings
            if process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
                successful_events += 1
            else:
                failed_events += 1
                # Remove incomplete embedding directory
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir)
                
        except Exception as e:
            print(f"\n✗ Error loading data for {event_name}: {e}")
            failed_events += 1
            # Remove incomplete embedding directory
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            continue
    
    return successful_events, failed_events

def main():
    """Create embeddings for all positive and negative samples using CopernicusFM"""
    print("Starting Embedding Generation with CopernicusFM")
    print("="*70)
    
    # Setup device
    device = utils.setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"Using device: {device}\n")
    
    # Load CopernicusFM model
    print("Loading CopernicusFM model...")
    encoder = load_model(device)
    print("✓ Model loaded\n")
    
    total_successful = 0
    total_failed = 0
    
    # Process positive samples
    pos_success, pos_fail = process_sample_type('positive', encoder, device)
    total_successful += pos_success
    total_failed += pos_fail
    
    # Process negative samples
    neg_success, neg_fail = process_sample_type('negative', encoder, device)
    total_successful += neg_success
    total_failed += neg_fail
    
    # Final summary
    print(f"\n{'='*70}")
    print("Embedding generation complete!")
    print(f"{'='*70}")
    print(f"POSITIVE samples - ✓ Success: {pos_success}, ✗ Failed: {pos_fail}")
    print(f"NEGATIVE samples - ✓ Success: {neg_success}, ✗ Failed: {neg_fail}")
    print(f"TOTAL            - ✓ Success: {total_successful}, ✗ Failed: {total_failed}")
    print(f"\nOutput: data/processed/embeddings/{{positive,negative}}/event_*/")

if __name__ == "__main__":
    main()