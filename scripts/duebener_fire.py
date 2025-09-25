from datetime import datetime

import matplotlib.pyplot as plt
import rootutils
from sklearn.metrics.pairwise import cosine_similarity

from burned_embedder import utils
from burned_embedder.analysis import analyze_fire_detection, compare_modes
from burned_embedder.data import find_closest_timestamps, load_s1, load_s2
from burned_embedder.model import load_model, process_embeddings
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
EDGE_SIZE = 128
RESOLUTION = 10
KERNEL_SIZE = 16
AREA = (16 * 10 / 1000) ** 2  # Surface area of one patch in km²


# Spectral parameters
S2_WAVELENGTHS = [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1610, 2190]
S2_BANDWIDTHS = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]
S1_WAVELENGTHS = [5e7, 5e7]  
S1_BANDWIDTHS = [1e9, 1e9]


def main():
    """Main analysis pipeline"""
    print("Starting Dueben Fire Detection Analysis")
    print("=" * 50)
    
    # Load data
    da_s1 = load_s1(LAT, LON)
    da_s2 = load_s2(LAT, LON)

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
        spectral_params = {
            's1_wavelengths': S1_WAVELENGTHS,
            's1_bandwidths': S1_BANDWIDTHS,
            's2_wavelengths': S2_WAVELENGTHS,
            's2_bandwidths': S2_BANDWIDTHS
        }
        embeddings, ndvi, dates = process_embeddings(encoder, da_s1, da_s2, timestamp_pairs, mode, 
                                                    device=device, lon=LON, lat=LAT, area=AREA, 
                                                    kernel_size=KERNEL_SIZE, **spectral_params)
        
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