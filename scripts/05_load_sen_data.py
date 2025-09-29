import sys
import rootutils
import pandas as pd
from tqdm import tqdm
import shutil
from pathlib import Path
from burned_embedder.data import load_s1_filtered, clean_metadata_nc, load_s2, calculate_search_dates

root_path = rootutils.find_root()


def download_satellite_data_for_continent(continent_name, deforest_df, observations_before=1, 
                                         observations_after=1, buffer_months=1, sample_type='positive'):
    """Download satellite data for all deforestation events in a continent."""
    print(f"\nProcessing {continent_name} - {sample_type.upper()} samples")
    print(f"Loaded {len(deforest_df)} {sample_type} records")
    
    for csv_row_idx, row in tqdm(deforest_df.iterrows(), total=len(deforest_df), 
                                 desc=f"{continent_name} {sample_type}"):
        deforest_data_dir = root_path / "data" / "processed" / "sen_data" / continent_name / sample_type / f"event_{csv_row_idx}"
        
        # Skip if data already exists
        s1_file = deforest_data_dir / "da_s1.nc"
        if s1_file.exists():
            continue
            
        lat = row['centroid_y']
        lon = row['centroid_x']
        earliest_alert = row['earliest_alert']
        latest_alert = row['latest_alert']

        # Try with original buffer first, then retry with larger buffer if needed
        for attempt, current_buffer in enumerate([buffer_months, buffer_months * 3], start=1):
            search_start_date, search_end_date = calculate_search_dates(
                earliest_alert, latest_alert, current_buffer
            )

            metadata = {
                'event_id': csv_row_idx,
                'continent': continent_name,
                'centroid_lat': row['centroid_y'],
                'centroid_lon': row['centroid_x'],
                'earliest_alert': str(row['earliest_alert']),
                'latest_alert': str(row['latest_alert']),
                'tile_name': row['tile_name'],
                'search_start_date': search_start_date,
                'search_end_date': search_end_date,
                'buffer_months': current_buffer,
                'sample_type': sample_type
            }
            
            if sample_type == 'negative':
                metadata["forest_fraction"] = row['forest_fraction']
                metadata["duration_days"] = row['positive_duration_days']

            if sample_type == 'positive':
                metadata['size_pixels'] = row['size_pixels']
                metadata['date_raster_file'] = row['date_raster_file']
                metadata['duration_days'] = row['duration_days']


            deforest_data_dir.mkdir(parents=True, exist_ok=True)

            try:
                da_s1 = load_s1_filtered(
                    lat, lon, search_start_date, search_end_date, 
                    earliest_alert, latest_alert,
                    observations_before, observations_after
                )

                if da_s1 is not None and len(da_s1.time) > 1:
                    s1_metadata = metadata.copy()
                    s1_metadata['satellite'] = 'Sentinel-1'
                    s1_metadata['processing_type'] = 'closest_observations'
                    s1_metadata['observations_before_deforestation'] = observations_before
                    s1_metadata['observations_after_deforestation'] = observations_after
                    if attempt > 1:
                        s1_metadata['retry_attempt'] = attempt
                    
                    da_s1_clean = clean_metadata_nc(da_s1, s1_metadata)
                    da_s1_clean.to_netcdf(s1_file)
                    break
                else:
                    if attempt == 1:
                        timesteps_found = len(da_s1.time) if da_s1 is not None else 0
                        print(f"\n⟳ Event {csv_row_idx}: Only {timesteps_found} timesteps, retrying with buffer={buffer_months * 3} months")
                        if deforest_data_dir.exists():
                            shutil.rmtree(deforest_data_dir)
                        continue
                    else:
                        print(f"\n✗ Event {csv_row_idx}: Still insufficient data after retry")
                        if deforest_data_dir.exists():
                            shutil.rmtree(deforest_data_dir)
                        break

            except Exception as e:
                print(f"\n✗ Error processing event {csv_row_idx} (attempt {attempt}): {e}")
                if deforest_data_dir.exists():
                    shutil.rmtree(deforest_data_dir)
                if attempt == 2:
                    break
                continue


def process_all_continents(observations_before=1, observations_after=1, buffer_months=1):
    """Process all three continents for both positive and negative samples."""
    
    continents = {
        # 'south_america': {
        #     'positive': root_path / "data/processed/radd/south_america_combined_clean.parquet",
        #     'negative': root_path / "data/processed/radd/south_america_negative_clean.parquet"
        # },
        'africa': {
            'positive': root_path / "data/processed/radd/africa_combined_clean.parquet",
            'negative': root_path / "data/processed/radd/africa_negative_clean.parquet"
        },
        'southeast_asia': {
            'positive': root_path / "data/processed/radd/southeast_asia_combined_clean.parquet",
            'negative': root_path / "data/processed/radd/southeast_asia_negative_clean.parquet"
        }
    }
    
    print("="*60)
    print("Starting Satellite Data Download for All Continents")
    print(f"Config: obs_before={observations_before}, obs_after={observations_after}, buffer={buffer_months} months")
    print("="*60)
    
    summary = {}
    
    for continent_name, paths in continents.items():
        print(f"\n{'='*60}")
        print(f"Processing {continent_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        continent_summary = {}
        
        # Process positive samples
        try:
            deforest_df_pos = pd.read_parquet(paths['positive'])
            continent_summary['positive_count'] = len(deforest_df_pos)
            
            download_satellite_data_for_continent(
                continent_name=continent_name,
                deforest_df=deforest_df_pos,
                observations_before=observations_before,
                observations_after=observations_after,
                buffer_months=buffer_months,
                sample_type='positive'
            )
        except Exception as e:
            print(f"Error processing positive samples for {continent_name}: {e}")
            continent_summary['positive_count'] = 0
        
        # Process negative samples
        try:
            deforest_df_neg = pd.read_parquet(paths['negative'])
            continent_summary['negative_count'] = len(deforest_df_neg)
            
            download_satellite_data_for_continent(
                continent_name=continent_name,
                deforest_df=deforest_df_neg,
                observations_before=observations_before,
                observations_after=observations_after,
                buffer_months=buffer_months,
                sample_type='negative'
            )
        except Exception as e:
            print(f"Error processing negative samples for {continent_name}: {e}")
            continent_summary['negative_count'] = 0
        
        summary[continent_name] = continent_summary
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for continent, counts in summary.items():
        print(f"{continent.replace('_', ' ').title()}:")
        print(f"  Positive samples: {counts.get('positive_count', 0)}")
        print(f"  Negative samples: {counts.get('negative_count', 0)}")
        print()


if __name__ == "__main__":
    
    process_all_continents(
        observations_before=1,
        observations_after=1,
        buffer_months=1
    )
    
    print("\nAll continents processed!")