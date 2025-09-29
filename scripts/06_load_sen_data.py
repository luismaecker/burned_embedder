import sys
import rootutils
import pandas as pd
from burned_embedder.data import load_s1_filtered, clean_metadata_nc, load_s2, calculate_search_dates

root_path = rootutils.find_root()


def main(observations_before=1, observations_after=1, buffer_months=1):
    """Download satellite data for all deforestation events with configurable observation windows."""
    print("Starting Deforestation Data Download")
    print(f"Observations before: {observations_before}, after: {observations_after}")
    print(f"Search buffer: {buffer_months} months around deforestation period")
    print("=" * 50)

    # Load deforestation data
    deforest_df = pd.read_parquet("data/interim/radd/south_america_combined.parquet")

    # Reset index
    deforest_df = deforest_df.reset_index(drop=True)    

    # Write cleaned df 
    deforest_df.to_parquet("data/processed/radd/south_america_combined_clean.parquet", index=False)
 

    # Load samples without deforestation
    negative_df = pd.read_parquet("data/processed/radd/negative_samples_sequential.parquet")


    print(f"Loaded {len(deforest_df)} deforestation records")
    
    for csv_row_idx, row in deforest_df.iterrows():
        print(f"\nProcessing event {csv_row_idx + 1}/{len(deforest_df)} (row index: {csv_row_idx})")
        
        lat = row['centroid_y']
        lon = row['centroid_x']
        earliest_alert = row['earliest_alert']
        latest_alert = row['latest_alert']

        # Calculate dynamic search dates for this event
        search_start_date, search_end_date = calculate_search_dates(
            earliest_alert, latest_alert, buffer_months
        )
        
        print(f"  Deforestation period: {earliest_alert} to {latest_alert}")
        print(f"  Search period: {search_start_date} to {search_end_date}")

        metadata = {
            'event_id': csv_row_idx,
            'centroid_lat': row['centroid_y'],
            'centroid_lon': row['centroid_x'],
            'earliest_alert': str(row['earliest_alert']),
            'latest_alert': str(row['latest_alert']),
            'duration_days': row['duration_days'],
            'size_pixels': row['size_pixels'],
            'tile_name': row['tile_name'],
            'date_raster_file': row['date_raster_file'],
            'search_start_date': search_start_date,
            'search_end_date': search_end_date,
            'buffer_months': buffer_months
        }

        deforest_data_dir = root_path / "data" / "processed" / "sen_data" / f"event_{csv_row_idx}"
        deforest_data_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load filtered Sentinel-1 data (closest observations to alerts)
            print(f"  Loading Sentinel-1 data with {observations_before} before + {observations_after} after...")
            da_s1 = load_s1_filtered(
                lat, lon, search_start_date, search_end_date, 
                earliest_alert, latest_alert,
                observations_before, observations_after
            )

            if da_s1 is not None:
                s1_metadata = metadata.copy()
                s1_metadata['satellite'] = 'Sentinel-1'
                s1_metadata['processing_type'] = 'closest_observations'
                s1_metadata['observations_before_deforestation'] = observations_before
                s1_metadata['observations_after_deforestation'] = observations_after
                
                da_s1_clean = clean_metadata_nc(da_s1, s1_metadata)
                s1_file = deforest_data_dir / "da_s1.nc"
                da_s1_clean.to_netcdf(s1_file)
                print(f"  ✓ S1 data saved: {s1_file} ({len(da_s1.time)} observations)")
            else:
                print(f"  ✗ No suitable S1 observations found")
                continue

            # Load all available Sentinel-2 data in the search period
            print(f"  Loading all available Sentinel-2 data...")
            da_s2 = load_s2(lat, lon, search_start_date, search_end_date, edge_size=100,
                            bands=["B02","B03","B04"], max_cloud_cover=100)

            if da_s2 is not None:
                s2_metadata = metadata.copy()
                s2_metadata['satellite'] = 'Sentinel-2'
                s2_metadata['processing_type'] = 'all_available_observations'
                s2_metadata['bands'] = 'B02,B03,B04'
                s2_metadata['max_cloud_cover'] = 100
                
                da_s2_clean = clean_metadata_nc(da_s2, s2_metadata)
                s2_file = deforest_data_dir / "da_s2.nc"
                da_s2_clean.to_netcdf(s2_file)
                print(f"  ✓ S2 data saved: {s2_file} ({len(da_s2.time)} observations)")
            else:
                print(f"  ✗ No suitable S2 observations found")
                
        except Exception as e:
            print(f"  ✗ Error processing event {csv_row_idx}: {e}")
            continue
    
    print(f"\nData download complete!")

if __name__ == "__main__":
    # You can adjust the buffer months here (default 1 month)
    main(observations_before=1, observations_after=1, buffer_months=1)