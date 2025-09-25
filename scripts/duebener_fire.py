from datetime import datetime
import sys

import numpy as np
import rootutils

from burned_embedder import utils
from burned_embedder.data import find_closest_timestamps, load_s1, load_s2
from burned_embedder.model import load_model, process_embeddings_batched

# Add TerraFM to Python path
# TODO: Update this path to where you put terrafm.py

sys.path.append('models/terrafm_models')

root_path = rootutils.find_root()

# Setup
device = utils.setup_device(gpu_index=1, memory_fraction=1.0)
print(f"Using device: {device}")

# Create output directory
embeddings_dir = root_path / "data" / "embeddings" / "dueben_fire_notime_2_terrafm"
embeddings_dir.mkdir(parents=True, exist_ok=True)

# Constants
LON = 13.29234188469853
LAT = 51.41003755616707
FIRE_DATE = datetime(2025, 7, 3)

start_date = "2017-01-01"
end_date = "2025-12-31"

def main():
    """Generate and save embeddings for all modes using TerraFM"""
    print("Starting Dueben Fire Embedding Generation with TerraFM")
    print("=" * 50)
    
    # Load data
    da_s1 = load_s1(LAT, LON, start_date=start_date, end_date=end_date).compute()
    da_s2 = load_s2(LAT, LON, start_date=start_date, end_date=end_date).compute()

    # Find matching timestamps
    timestamp_pairs = find_closest_timestamps(da_s1.time.values, da_s2.time.values, max_diff_days=2)
    print(f"Found {len(timestamp_pairs)} matching S1-S2 pairs")
    
    if len(timestamp_pairs) == 0:
        print("No matching timestamps found. Exiting.")
        return
    
    # Load TerraFM model
    # TODO: Update this path to point to your downloaded TerraFM weights
    weights_path = "models/terrafm_models/TerraFM-B.pth"  # Update this path
    encoder = load_model(device, model_size='base', weights_path=weights_path)
    
    # Process embeddings for each mode
    modes = ['s1_only', 's2_only']
    
    for mode in modes:
        print(f"\n{'='*20} Processing {mode.upper()} Mode {'='*20}")
        
        # Process embeddings
        embeddings, ndvi, dates = process_embeddings_batched(
            encoder, da_s1, da_s2, timestamp_pairs, 
            mode=mode, device=device, lon=LON, lat=LAT
        )
        
        # Save embeddings and metadata
        np.save(embeddings_dir / f"{mode}_embeddings.npy", embeddings)
        np.save(embeddings_dir / f"{mode}_ndvi.npy", ndvi)
        np.save(embeddings_dir / f"{mode}_dates.npy", dates)
        
        print(f"Saved {len(embeddings)} embeddings for {mode} mode")
        print(f"Embedding shape: {embeddings.shape}")
    
    # Save metadata
    metadata = {
        'lon': LON,
        'lat': LAT,
        'fire_date': FIRE_DATE,
        'start_date': start_date,
        'end_date': end_date,
        'num_timestamp_pairs': len(timestamp_pairs)
    }
    
    np.save(embeddings_dir / "metadata.npy", metadata)
    
    print(f"\nEmbedding generation complete! All files saved to: {embeddings_dir}")
    print(f"Modes processed: {', '.join([m.upper() for m in modes])}")

if __name__ == "__main__":
    main()
        
#         # Analyze fire detection
#         results[mode] = analyze_fire_detection(embeddings, dates, ndvi, mode, mode_dir, FIRE_DATE)
        
#         # Create similarity matrix
#         if len(embeddings) > 1:
#             similarity_matrix = cosine_similarity(embeddings)
            
#             plt.figure(figsize=(10, 8))
#             plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
#             plt.colorbar(label='Cosine Similarity')
#             plt.title(f'{mode.upper()}: Embedding Similarity Between Timesteps')
#             plt.xlabel('Timestep')
#             plt.ylabel('Timestep')
#             plt.savefig(mode_dir / f'{mode}_similarity_matrix.png', dpi=300, bbox_inches='tight')
#             plt.show()
    
#     # Compare modes
#     print(f"\n{'='*20} Comparing All Modes {'='*20}")
#     compare_modes(results, figures_dir, FIRE_DATE)
    
#     # Create summary visualization showing all three modes together
#     create_summary_visualization(results, figures_dir, FIRE_DATE, LAT, LON)
    
#     print(f"\nAnalysis complete! All figures saved to: {figures_dir}")
#     print(f"Modes analyzed: {', '.join([m.upper() for m in modes])}")
