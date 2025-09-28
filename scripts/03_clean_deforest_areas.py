import pandas as pd
import rootutils

root_path = rootutils.find_root()

deforest_df = pd.read_parquet(root_path / "data/interim/radd/south_america_combined.parquet")
deforest_df = deforest_df[deforest_df['duration_days'] <= 30]
deforest_df = deforest_df[deforest_df['size_pixels'] >= 100]
deforest_df.reset_index(drop=True, inplace=True)
deforest_df = deforest_df[deforest_df['tile_name'] != '10S_050W_radd_alerts']
deforest_df.to_parquet(root_path / "data/processed/radd/south_america_combined_clean.parquet")

print(f"Cleaned deforestation areas: {len(deforest_df)} entries")