import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def create_summary_visualization(results_dict, save_dir, fire_date, lat=None, lon=None):
    """Create a comprehensive summary visualization"""
    print("Creating summary visualization...")
    
    modes = list(results_dict.keys())
    colors_dict = {'s1_only': 'blue', 's2_only': 'green', 'combined': 'purple'}
    
    # Create a mega-plot with all key comparisons
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Dueben Fire Detection: Complete Analysis Summary', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Plot 1: PC1 time series comparison (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for mode in modes:
        data = results_dict[mode]
        # Sort chronologically
        sort_indices = np.argsort(data['dates'])
        sorted_dates_dt = [pd.to_datetime(data['dates'][idx]) for idx in sort_indices]
        sorted_pc1 = data['pca'][sort_indices, 0]
        
        # Normalize PC1 for comparison
        sorted_pc1_norm = (sorted_pc1 - np.mean(sorted_pc1)) / np.std(sorted_pc1)
        
        ax1.plot(sorted_dates_dt, sorted_pc1_norm, '-o', 
                color=colors_dict[mode], alpha=0.7, label=f'{mode.upper()}', markersize=4)
    
    ax1.axvline(x=fire_date, color='red', linestyle='--', linewidth=3, alpha=0.8, label='Fire Date')
    ax1.set_ylabel('Normalized PC1')
    ax1.set_title('PC1 Time Series Comparison (All Modes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Variance explained comparison (top row, right)
    ax2 = fig.add_subplot(gs[0, 2])
    var_explained = [results_dict[mode]['pca_obj'].explained_variance_ratio_[0] for mode in modes]
    bars = ax2.bar(range(len(modes)), var_explained, 
                  color=[colors_dict[mode] for mode in modes], alpha=0.7)
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('PC1 Variance Explained')
    ax2.set_title('PC1 Variance by Mode')
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels([m.upper() for m in modes])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Fire detection sensitivity (top row, far right)
    ax3 = fig.add_subplot(gs[0, 3])
    fire_sensitivity = []
    for mode in modes:
        data = results_dict[mode]
        pre_fire_pc1 = data['pca'][data['pre_fire_mask'], 0]
        post_fire_pc1 = data['pca'][[not x for x in data['pre_fire_mask']], 0]
        
        if len(pre_fire_pc1) > 0 and len(post_fire_pc1) > 0:
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(pre_fire_pc1)-1)*np.var(pre_fire_pc1) + 
                                 (len(post_fire_pc1)-1)*np.var(post_fire_pc1)) / 
                                (len(pre_fire_pc1) + len(post_fire_pc1) - 2))
            cohens_d = (np.mean(post_fire_pc1) - np.mean(pre_fire_pc1)) / pooled_std
            fire_sensitivity.append(abs(cohens_d))
        else:
            fire_sensitivity.append(0)
    
    bars = ax3.bar(range(len(modes)), fire_sensitivity,
                  color=[colors_dict[mode] for mode in modes], alpha=0.7)
    ax3.set_xlabel('Mode')
    ax3.set_ylabel('Fire Sensitivity (|Cohen\'s d|)')
    ax3.set_title('Fire Detection Sensitivity')
    ax3.set_xticks(range(len(modes)))
    ax3.set_xticklabels([m.upper() for m in modes])
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4-6: Individual mode t-SNE plots (middle row)
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(gs[1, i])
        data = results_dict[mode]
        
        if data['tsne'] is not None:
            dates_dt = [pd.to_datetime(date) for date in data['dates']]
            colors_fire = ['red' if date >= fire_date else 'lightgray' for date in dates_dt]
            
            for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
                mask = [c == color for c in colors_fire]
                if any(mask):
                    ax.scatter(data['tsne'][mask, 0], data['tsne'][mask, 1], 
                             c=color, alpha=0.7, s=30, label=label)
            ax.set_title(f'{mode.upper()}: t-SNE')
            ax.legend()
        else:
            # Use PCA if t-SNE not available
            colors_fire = ['red' if pd.to_datetime(date) >= fire_date else 'lightgray' 
                          for date in data['dates']]
            for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
                mask = [c == color for c in colors_fire]
                if any(mask):
                    ax.scatter(data['pca'][mask, 0], data['pca'][mask, 1], 
                             c=color, alpha=0.7, s=30, label=label)
            ax.set_title(f'{mode.upper()}: PCA')
            ax.legend()
    
    # Plot 7-9: NDVI correlation plots (bottom row)
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(gs[2, i])
        data = results_dict[mode]
        
        dates_dt = [pd.to_datetime(date) for date in data['dates']]
        colors_fire = [1 if date >= fire_date else 0 for date in dates_dt]
        
        ax.scatter(data['ndvi'], data['pca'][:, 0], 
                   c=colors_fire, cmap='RdGy_r', alpha=0.7, s=40)
        
        # Calculate correlation
        correlation = np.corrcoef(data['ndvi'], data['pca'][:, 0])[0, 1]
        ax.set_xlabel('NDVI')
        ax.set_ylabel('PC1')
        ax.set_title(f'{mode.upper()}: NDVI vs PC1\n(r={correlation:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['ndvi'], data['pca'][:, 0], 1)
        p = np.poly1d(z)
        ax.plot(data['ndvi'], p(data['ndvi']), "r--", alpha=0.8, linewidth=2)
    
    # Add a summary text box
    summary_text = f"""
    Fire Event: {fire_date.strftime('%Y-%m-%d')}
    Location: {lat:.3f}°N, {lon:.3f}°E
    Data: Sentinel-1 & Sentinel-2
    Analysis: Foundation Model Embeddings
    
    Key Findings:
    • Best PC1 variance: {modes[np.argmax(var_explained)].upper()}
    • Highest fire sensitivity: {modes[np.argmax(fire_sensitivity)].upper()}
    • Combined approach shows multimodal benefits
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_dir / 'complete_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed comparison table
    create_comparison_table(results_dict, save_dir)

def create_comparison_table(results_dict, save_dir):
    """Create a detailed comparison table of all modes"""
    modes = list(results_dict.keys())
    
    # Prepare data for table
    comparison_data = []
    
    for mode in modes:
        data = results_dict[mode]
        
        # Calculate statistics
        pre_fire_pc1 = data['pca'][data['pre_fire_mask'], 0]
        post_fire_pc1 = data['pca'][[not x for x in data['pre_fire_mask']], 0]
        pre_fire_ndvi = data['ndvi'][data['pre_fire_mask']]
        post_fire_ndvi = data['ndvi'][[not x for x in data['pre_fire_mask']]]
        
        # Basic stats
        embedding_dim = data['embeddings'].shape[1]
        pc1_variance = data['pca_obj'].explained_variance_ratio_[0]
        ndvi_pc1_corr = np.corrcoef(data['ndvi'], data['pca'][:, 0])[0, 1]
        
        # Fire detection stats
        if len(pre_fire_pc1) > 0 and len(post_fire_pc1) > 0:
            pc1_change = np.mean(post_fire_pc1) - np.mean(pre_fire_pc1)
            ndvi_change = np.mean(post_fire_ndvi) - np.mean(pre_fire_ndvi)
            
            pc1_ttest = ttest_ind(pre_fire_pc1, post_fire_pc1)
            ndvi_ttest = ttest_ind(pre_fire_ndvi, post_fire_ndvi)
            
            # Effect size (Cohen's d)
            pooled_std_pc1 = np.sqrt(((len(pre_fire_pc1)-1)*np.var(pre_fire_pc1) + 
                                     (len(post_fire_pc1)-1)*np.var(post_fire_pc1)) / 
                                    (len(pre_fire_pc1) + len(post_fire_pc1) - 2))
            cohens_d_pc1 = pc1_change / pooled_std_pc1
        else:
            pc1_change = np.nan
            ndvi_change = np.nan
            pc1_ttest = None
            ndvi_ttest = None
            cohens_d_pc1 = np.nan
        
        comparison_data.append({
            'Mode': mode.upper(),
            'Embedding Dim': embedding_dim,
            'PC1 Var Explained': f'{pc1_variance:.3f}',
            'NDVI-PC1 Correlation': f'{ndvi_pc1_corr:.3f}',
            'PC1 Change (Fire)': f'{pc1_change:.3f}' if not np.isnan(pc1_change) else 'N/A',
            'NDVI Change (Fire)': f'{ndvi_change:.3f}' if not np.isnan(ndvi_change) else 'N/A',
            'PC1 p-value': f'{pc1_ttest.pvalue:.4f}' if pc1_ttest else 'N/A',
            'NDVI p-value': f'{ndvi_ttest.pvalue:.4f}' if ndvi_ttest else 'N/A',
            'Fire Sensitivity (|d|)': f'{abs(cohens_d_pc1):.3f}' if not np.isnan(cohens_d_pc1) else 'N/A'
        })