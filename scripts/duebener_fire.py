from datetime import datetime

import matplotlib.pyplot as plt
import rootutils
from sklearn.metrics.pairwise import cosine_similarity

from burned_embedder import utils
from burned_embedder.analysis import analyze_fire_detection, compare_modes
from burned_embedder.data import find_closest_timestamps, load_s1, load_s2
from burned_embedder.model import load_model, process_embeddings_batched
from burned_embedder.plot import create_summary_visualization

root_path = rootutils.find_root()

# Setup
plt.style.use('seaborn-v0_8')
device = utils.setup_device(gpu_index=1, memory_fraction=1.0)
print(f"Using device: {device}")

# Create output directories
figures_dir = root_path / "reports" / "figures" / "dueben_fire"
figures_dir.mkdir(parents=True, exist_ok=True)

# Constants
LON = 13.32200476
LAT = 51.4222985
FIRE_DATE = datetime(2025, 7, 3)

start_date = "2017-01-01"
end_date = "2025-12-31"


def main():
    """Main analysis pipeline"""
    print("Starting Dueben Fire Detection Analysis")
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
    
    # Load model
    encoder = load_model(device)
    
    # Process embeddings for each mode
    modes = ['s1_only', 's2_only', 'combined']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*20} Processing {mode.upper()} Mode {'='*20}")
        
        # Create mode-specific directory
        mode_dir = figures_dir / mode
        mode_dir.mkdir(exist_ok=True)
        
        # Process embeddings
        embeddings, ndvi, dates = process_embeddings_batched(encoder, da_s1, da_s2, timestamp_pairs, 
                                                            mode=mode, device=device, lon=LON, lat=LAT)
        
        # Analyze fire detection
        results[mode] = analyze_fire_detection(embeddings, dates, ndvi, mode, mode_dir, FIRE_DATE)
        
        # Create similarity matrix
        if len(embeddings) > 1:
            similarity_matrix = cosine_similarity(embeddings)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Cosine Similarity')
            plt.title(f'{mode.upper()}: Embedding Similarity Between Timesteps')
            plt.xlabel('Timestep')
            plt.ylabel('Timestep')
            plt.savefig(mode_dir / f'{mode}_similarity_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Compare modes
    print(f"\n{'='*20} Comparing All Modes {'='*20}")
    compare_modes(results, figures_dir, FIRE_DATE)
    
    # Create summary visualization showing all three modes together
    create_summary_visualization(results, figures_dir, FIRE_DATE, LAT, LON)
    
    print(f"\nAnalysis complete! All figures saved to: {figures_dir}")
    print(f"Modes analyzed: {', '.join([m.upper() for m in modes])}")

if __name__ == "__main__":
    main()