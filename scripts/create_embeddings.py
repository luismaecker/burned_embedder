import numpy as np
import rootutils
import xarray as xr
from tqdm import tqdm

from burned_embedder import utils
from burned_embedder.model import load_model, process_batch_unified

root_path = rootutils.find_root()

def process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
    """Process embeddings for a single deforestation event with 1 or 2 timestamps using batch processing"""
    print(f"  Processing event {event_id}")
    
    # Extract metadata from the dataset
    centroid_lat = float(da_s1.attrs.get('centroid_lat', 0))
    centroid_lon = float(da_s1.attrs.get('centroid_lon', 0))
    
    # Handle different numbers of timesteps
    num_timesteps = len(da_s1.time)
    if num_timesteps == 0:
        print(f"  ⚠️  Event {event_id} has no timesteps. Skipping.")
        return False
    elif num_timesteps == 1:
        print(f"  ⚠️  Event {event_id} has only 1 timestep, expected 2. Skipping.")
        return False
    elif num_timesteps > 2:
        print(f"  ⚠️  Event {event_id} has more than 2 timesteps, expected 2. Skipping.")
        return False

    
    timestamps = da_s1.time.values
    
    try:
        import pandas as pd
        
        # Normal case with 2 timesteps
        da_s1_batch = [da_s1.isel(time=0), da_s1.isel(time=1)]  # Both timesteps
        da_s2_batch = [None, None]  # Not used for S1-only mode
        
        # Convert timestamps to datetime
        acquisition_dates_batch = [
            pd.to_datetime(timestamps[0]).to_pydatetime(),
            pd.to_datetime(timestamps[1]).to_pydatetime()
        ]
        
        # Process both embeddings in a single batch call - much more efficient!
        embeddings = process_batch_unified(
            encoder, da_s1_batch, da_s2_batch, acquisition_dates_batch,
            mode='s1_only', device=device, 
            lon=centroid_lon, lat=centroid_lat
        )
        
        # Save embeddings separately
        embedd_before = embeddings[0]  # First embedding (before)
        embedd_after = embeddings[1]   # Second embedding (after)
        
        np.save(output_dir / "embedd_before.npy", embedd_before)
        np.save(output_dir / "embedd_after.npy", embedd_after)
        
        print(f"    ✓ Saved both embeddings in batch: {embeddings.shape}")
        
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
            'size_pixels': da_s1.attrs.get('size_pixels', ''),
            'model_mode': 's1_only'
        }
        
        np.save(output_dir / "metadata.npy", metadata)
        print("    ✓ Saved metadata")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing event {event_id}: {e}")
        return False

def main():
    """Create embeddings for all deforestation events using CopernicusFM"""
    print("Starting Deforestation Event Embedding Generation with CopernicusFM")
    print("=" * 70)
    
    # Setup device
    device = utils.setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"Using device: {device}")
    
    # Load CopernicusFM model
    encoder = load_model(device)
    
    # Find all deforestation events
    sen_data_dir = root_path / "data" / "processed" / "sen_data"
    event_dirs = sorted([d for d in sen_data_dir.iterdir() if d.is_dir() and d.name.startswith('event_')])
    
    print(f"Found {len(event_dirs)} deforestation events")
    
    # Create output directory
    embeddings_base_dir = root_path / "data" / "processed" / "embeddings"
    embeddings_base_dir.mkdir(parents=True, exist_ok=True)
    
    successful_events = 0
    failed_events = 0
    
    # Process each event
    for event_dir in tqdm(event_dirs, desc="Processing events"):
        event_name = event_dir.name
        event_id = int(event_name.split('_')[1])
        
        # Check if S1 data exists
        s1_file = event_dir / "da_s1.nc"
        if not s1_file.exists():
            print(f"⚠️  S1 file not found for {event_name}: {s1_file}")
            failed_events += 1
            continue
        
        # Create output directory for this event
        output_dir = embeddings_base_dir / event_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if embeddings already exist
        if (output_dir / "embedd_before.npy").exists() and (output_dir / "embedd_after.npy").exists():
            print(f"  ⏭️  Embeddings already exist for {event_name}, skipping")
            continue
        
        try:
            # Load S1 data
            da_s1 = xr.open_dataset(s1_file)['sentinel-1-rtc']
            
            # Process embeddings
            if process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
                successful_events += 1
            else:
                failed_events += 1
                # Remove the embedding directory if processing failed
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir)
                    print(f"    🗑️  Removed incomplete embedding directory: {output_dir.name}")
                
        except Exception as e:
            print(f"  ✗ Error loading data for {event_name}: {e}")
            failed_events += 1
            # Remove the embedding directory if there was an error
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
                print(f"    🗑️  Removed incomplete embedding directory: {output_dir.name}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("Embedding generation complete!")
    print(f"  ✓ Successfully processed: {successful_events} events")
    print(f"  ✗ Failed: {failed_events} events")
    print(f"  📁 Output directory: {embeddings_base_dir}")
    
    if successful_events > 0:
        print("\nEmbeddings saved as:")
        print("  - embedd_before.npy (before deforestation)")
        print("  - embedd_after.npy (after deforestation)")
        print("  - metadata.npy (event metadata)")

if __name__ == "__main__":
    main()
