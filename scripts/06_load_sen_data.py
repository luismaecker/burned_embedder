import sys
import rootutils
import pandas as pd
from tqdm import tqdm
import shutil
from pathlib import Path
from burned_embedder.data import load_s1_filtered, clean_metadata_nc, load_s2, calculate_search_dates

root_path = rootutils.find_root()

#TODO: delete old
# def main(observations_before=1, observations_after=1, buffer_months=1, sample_type='positive'):
#     """Download satellite data for all deforestation events with configurable observation windows."""
#     print(f"Starting {sample_type.upper()} Sample Data Download")
#     print(f"Config: obs_before={observations_before}, obs_after={observations_after}, buffer={buffer_months} months")

#     # Load sample data based on type
#     if sample_type == 'positive':
#         deforest_df = pd.read_parquet("data/processed/radd/south_america_combined_clean_sampled_15.parquet")
#     elif sample_type == 'negative':
#         deforest_df = pd.read_parquet("data/processed/radd/negative_samples_sequential_sampled_15.parquet")
#     else:
#         raise ValueError(f"Invalid sample_type: {sample_type}. Must be 'positive' or 'negative'")

#     print(f"Loaded {len(deforest_df)} {sample_type} records\n")
    
#     for csv_row_idx, row in tqdm(deforest_df.iterrows(), total=len(deforest_df), desc=f"Processing {sample_type} samples"):
#         deforest_data_dir = root_path / "data" / "processed" / "sen_data_15" / sample_type / f"event_{csv_row_idx}"
        
#         # Skip if data already exists
#         s1_file = deforest_data_dir / "da_s1.nc"
#         if s1_file.exists():
#             continue
            
#         lat = row['centroid_y']
#         lon = row['centroid_x']
#         earliest_alert = row['earliest_alert']
#         latest_alert = row['latest_alert']

#         # Calculate dynamic search dates for this event
#         search_start_date, search_end_date = calculate_search_dates(
#             earliest_alert, latest_alert, buffer_months
#         )

#         metadata = {
#             'event_id': csv_row_idx,
#             'centroid_lat': row['centroid_y'],
#             'centroid_lon': row['centroid_x'],
#             'earliest_alert': str(row['earliest_alert']),
#             'latest_alert': str(row['latest_alert']),
#             'duration_days': row['duration_days'],
#             'tile_name': row['tile_name'],
#             'search_start_date': search_start_date,
#             'search_end_date': search_end_date,
#             'buffer_months': buffer_months,
#             'sample_type': sample_type
#         }
        
#         if sample_type == 'negative':
#             metadata["forest_fraction"] = row['forest_fraction']

#         # Add columns that only exist in positive samples
#         if sample_type == 'positive':
#             metadata['size_pixels'] = row['size_pixels']
#             metadata['date_raster_file'] = row['date_raster_file']

#         deforest_data_dir.mkdir(parents=True, exist_ok=True)

#         try:
#             # Load filtered Sentinel-1 data
#             da_s1 = load_s1_filtered(
#                 lat, lon, search_start_date, search_end_date, 
#                 earliest_alert, latest_alert,
#                 observations_before, observations_after
#             )

#             if da_s1 is not None:
#                 s1_metadata = metadata.copy()
#                 s1_metadata['satellite'] = 'Sentinel-1'
#                 s1_metadata['processing_type'] = 'closest_observations'
#                 s1_metadata['observations_before_deforestation'] = observations_before
#                 s1_metadata['observations_after_deforestation'] = observations_after
                
#                 da_s1_clean = clean_metadata_nc(da_s1, s1_metadata)
#                 da_s1_clean.to_netcdf(s1_file)
#             else:
#                 print(f"\n✗ Event {csv_row_idx}: No suitable S1 observations found")
#                 if deforest_data_dir.exists():
#                     shutil.rmtree(deforest_data_dir)
#                 continue

#             # # Load all available Sentinel-2 data
#             # da_s2 = load_s2(lat, lon, search_start_date, search_end_date, edge_size=100,
#             #                 bands=["B02","B03","B04"], max_cloud_cover=100)

#             # if da_s2 is not None:
#             #     s2_metadata = metadata.copy()
#             #     s2_metadata['satellite'] = 'Sentinel-2'
#             #     s2_metadata['processing_type'] = 'all_available_observations'
#             #     s2_metadata['bands'] = 'B02,B03,B04'
#             #     s2_metadata['max_cloud_cover'] = 100
                
#             #     da_s2_clean = clean_metadata_nc(da_s2, s2_metadata)
#             #     s2_file = deforest_data_dir / "da_s2.nc"
#             #     da_s2_clean.to_netcdf(s2_file)
                
#         except Exception as e:
#             print(f"\n✗ Error processing event {csv_row_idx}: {e}")
#             if deforest_data_dir.exists():
#                 shutil.rmtree(deforest_data_dir)
#             continue
    
#     print(f"\n{sample_type.upper()} sample data download complete!")

def main(observations_before=1, observations_after=1, buffer_months=1, sample_type='positive'):
    """Download satellite data for all deforestation events with configurable observation windows."""
    print(f"Starting {sample_type.upper()} Sample Data Download")
    print(f"Config: obs_before={observations_before}, obs_after={observations_after}, buffer={buffer_months} months")

    # Load sample data based on type
    if sample_type == 'positive':
        deforest_df = pd.read_parquet("data/processed/radd/south_america_combined_clean_sampled_15.parquet")
    elif sample_type == 'negative':
        deforest_df = pd.read_parquet("data/processed/radd/negative_samples_sequential_sampled_15.parquet")
    else:
        raise ValueError(f"Invalid sample_type: {sample_type}. Must be 'positive' or 'negative'")

    print(f"Loaded {len(deforest_df)} {sample_type} records\n")
    
    for csv_row_idx, row in tqdm(deforest_df.iterrows(), total=len(deforest_df), desc=f"Processing {sample_type} samples"):
        deforest_data_dir = root_path / "data" / "processed" / "sen_data" / sample_type / f"event_{csv_row_idx}"
        
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
                'centroid_lat': row['centroid_y'],
                'centroid_lon': row['centroid_x'],
                'earliest_alert': str(row['earliest_alert']),
                'latest_alert': str(row['latest_alert']),
                'duration_days': row['duration_days'],
                'tile_name': row['tile_name'],
                'search_start_date': search_start_date,
                'search_end_date': search_end_date,
                'buffer_months': current_buffer,
                'sample_type': sample_type
            }
            
            if sample_type == 'negative':
                metadata["forest_fraction"] = row['forest_fraction']

            if sample_type == 'positive':
                metadata['size_pixels'] = row['size_pixels']
                metadata['date_raster_file'] = row['date_raster_file']

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
                    break  # Success, exit retry loop
                else:
                    if attempt == 1:
                        timesteps_found = len(da_s1.time) if da_s1 is not None else 0
                        print(f"\n⟳ Event {csv_row_idx}: Only {timesteps_found} timesteps found, retrying with buffer={buffer_months * 3} months")
                        if deforest_data_dir.exists():
                            shutil.rmtree(deforest_data_dir)
                        continue  # Retry with larger buffer
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
    
    print(f"\n{sample_type.upper()} sample data download complete!")

if __name__ == "__main__":

    # Process negative samples
    main(observations_before=1, observations_after=1, buffer_months=1, sample_type='negative')

    # Process positive samples
    main(observations_before=1, observations_after=1, buffer_months=1, sample_type='positive')