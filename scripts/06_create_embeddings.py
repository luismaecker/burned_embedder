import numpy as np
import rootutils
import xarray as xr
from tqdm import tqdm
from pathlib import Path

from burned_embedder import utils
from burned_embedder.model import load_copFM, process_batch_unified

root_path = rootutils.find_root()

def process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
    """Process embeddings for a single event with 2 timestamps using batch processing"""
    
    centroid_lat = float(da_s1.attrs.get('centroid_lat', 0))
    centroid_lon = float(da_s1.attrs.get('centroid_lon', 0))
    
    num_timesteps = len(da_s1.time)
    if num_timesteps != 2:
        print(f"\n✗ Event {event_id}: Expected 2 timesteps, got {num_timesteps}. Skipping.")
        return False
    
    timestamps = da_s1.time.values
    
    try:
        import pandas as pd
        
        da_s1_batch = [da_s1.isel(time=0), da_s1.isel(time=1)]
        
        acquisition_dates_batch = [
            pd.to_datetime(timestamps[0]).to_pydatetime(),
            pd.to_datetime(timestamps[1]).to_pydatetime()
        ]
        
        embeddings = process_batch_unified(
            encoder, da_s1_batch, acquisition_dates_batch, 
            device, centroid_lon, centroid_lat
        )
        
        embedd_before = embeddings[0]
        embedd_after = embeddings[1]
        
        np.save(output_dir / "embedd_before.npy", embedd_before)
        np.save(output_dir / "embedd_after.npy", embedd_after)
        
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
            'continent': da_s1.attrs.get('continent', ''),
            'model_mode': 's1_only'
        }
        
        if 'size_pixels' in da_s1.attrs:
            metadata['size_pixels'] = da_s1.attrs.get('size_pixels', '')
        
        np.save(output_dir / "metadata.npy", metadata)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing event {event_id}: {e}")
        return False

def process_continent_sample_type(continent_name, sample_type, encoder, device):
    """Process all events for a given continent and sample type"""
    print(f"\nProcessing {continent_name} - {sample_type} samples")
    
    sen_data_dir = root_path / "data" / "processed" / "sen_data" / continent_name / sample_type
    
    if not sen_data_dir.exists():
        print(f"  ✗ Directory not found: {sen_data_dir}")
        return 0, 0
    
    event_dirs = sorted([d for d in sen_data_dir.iterdir() if d.is_dir() and d.name.startswith('event_')])
    
    print(f"  Found {len(event_dirs)} events")
    
    embeddings_dir = root_path / "data" / "processed" / "embeddings" / continent_name / sample_type
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    successful_events = 0
    failed_events = 0
    
    for event_dir in tqdm(event_dirs, desc=f"  {continent_name}/{sample_type}", leave=False):
        event_name = event_dir.name
        event_id = int(event_name.split('_')[1])
        
        s1_file = event_dir / "da_s1.nc"
        if not s1_file.exists():
            failed_events += 1
            continue
        
        output_dir = embeddings_dir / event_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if (output_dir / "embedd_before.npy").exists() and (output_dir / "embedd_after.npy").exists():
            successful_events += 1
            continue
        
        try:
            da_s1 = xr.open_dataset(s1_file)['sentinel-1-rtc']
            
            if process_single_event_embeddings(encoder, da_s1, event_id, device, output_dir):
                successful_events += 1
            else:
                failed_events += 1
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir)
                
        except Exception as e:
            failed_events += 1
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            continue
    
    return successful_events, failed_events


def process_all_continents():
    """Create embeddings for all continents (positive and negative samples) using CopernicusFM"""
    print("="*70)
    print("Starting Embedding Generation with CopernicusFM")
    print("="*70)
    
    device = utils.setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"Using device: {device}\n")
    
    print("Loading CopernicusFM model...")
    encoder = load_copFM(device)
    print("✓ Model loaded\n")
    
    continents = ['south_america', 'africa', 'southeast_asia']
    sample_types = ['positive', 'negative']
    
    summary = {}
    
    for continent_name in continents:
        print(f"\n{'='*70}")
        print(f"Processing {continent_name.replace('_', ' ').title()}")
        print(f"{'='*70}")
        
        continent_summary = {}
        
        for sample_type in sample_types:
            success, fail = process_continent_sample_type(
                continent_name, sample_type, encoder, device
            )
            continent_summary[sample_type] = {
                'success': success,
                'failed': fail
            }
        
        summary[continent_name] = continent_summary
    
    print(f"\n{'='*70}")
    print("Embedding Generation Complete!")
    print(f"{'='*70}")
    
    for continent, results in summary.items():
        print(f"\n{continent.replace('_', ' ').title()}:")
        for sample_type, counts in results.items():
            print(f"  {sample_type.upper():8} - ✓ Success: {counts['success']:4}, ✗ Failed: {counts['failed']:4}")
    
    total_success = sum(results[st]['success'] for results in summary.values() for st in sample_types)
    total_failed = sum(results[st]['failed'] for results in summary.values() for st in sample_types)
    
    print(f"\nTOTAL: ✓ Success: {total_success}, ✗ Failed: {total_failed}")
    print(f"\nOutput: data/processed/embeddings/{{continent}}/{{positive,negative}}/event_*/")


if __name__ == "__main__":
    process_all_continents()