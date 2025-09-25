import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def calculate_ndvi(da_s2_t):
    """Calculate NDVI from Sentinel-2 data"""
    nir = da_s2_t.sel(band='B08').values  # Near-infrared
    red = da_s2_t.sel(band='B04').values  # Red
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.nanmean(ndvi)



def perform_dimensionality_reduction(embeddings_array):
    """Apply PCA and t-SNE to embeddings (assumes clean input)"""
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # PCA
    n_components = min(10, len(embeddings_array)-1, embeddings_array.shape[1])
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_array)
    
    # t-SNE
    has_tsne = False
    embeddings_tsne = None
    if len(embeddings_array) > 5:
        perplexity = min(5, len(embeddings_array)-1, 30)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_tsne = tsne.fit_transform(embeddings_array)
        has_tsne = True
    
    return pca, embeddings_pca, embeddings_tsne, has_tsne

def seasonal_curve(day, a, b, c, d):
    """Sinusoidal function to model seasonal variation"""
    return a * np.sin(2 * np.pi * day / 365 + b) + c

def analyze_fire_detection(embeddings_array, dates_array, ndvi_array, mode, save_dir, fire_date):
    """Comprehensive fire detection analysis"""
    print(f"Analyzing fire detection for {mode} mode...")
    
    pca, embeddings_pca, embeddings_tsne, has_tsne = perform_dimensionality_reduction(embeddings_array)
    
    # Convert dates
    dates_datetime = [pd.to_datetime(date) for date in dates_array]
    day_of_year = [date.timetuple().tm_yday for date in dates_datetime]
    fire_day_of_year = fire_date.timetuple().tm_yday
    
    # Create pre/post fire masks
    pre_fire_mask = [date < fire_date for date in dates_datetime]
    colors = ['red' if date >= fire_date else 'lightgray' for date in dates_datetime]
    
    # Sort data chronologically
    sort_indices = np.argsort(dates_array)
    sorted_dates = dates_array[sort_indices]
    sorted_embeddings_pca = embeddings_pca[sort_indices]
    sorted_ndvi = ndvi_array[sort_indices]
    sorted_dates_dt = [pd.to_datetime(date) for date in sorted_dates]
    
    # Create comprehensive plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'Fire Detection Analysis - {mode.upper()} Mode', fontsize=16, fontweight='bold')
    
    # Plot 1: PC1 over day of year with fire detection
    for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
        mask = [c == color for c in colors]
        if any(mask):
            axes[0,0].scatter(np.array(day_of_year)[mask], embeddings_pca[mask, 0], 
                           c=color, alpha=0.7, s=50, label=label)
    
    axes[0,0].axvline(x=fire_day_of_year, color='red', linestyle='--', linewidth=2, 
                   label=f'Fire: {fire_date.strftime("%Y-%m-%d")}')
    axes[0,0].set_xlabel('Day of Year')
    axes[0,0].set_ylabel('First Principal Component')
    axes[0,0].set_title(f'{mode.upper()}: PC1 Fire Detection')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: t-SNE or PCA 2D projection
    if has_tsne and embeddings_tsne is not None:
        for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
            mask = [c == color for c in colors]
            if any(mask):
                axes[0,1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                               c=color, alpha=0.7, s=50, label=label)
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        axes[0,1].set_title(f'{mode.upper()}: t-SNE Pre vs Post-fire')
    else:
        axes[0,1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                       c=[0 if c == 'lightgray' else 1 for c in colors], 
                       cmap='RdGy_r', alpha=0.7, s=50)
        axes[0,1].set_xlabel('PC1')
        axes[0,1].set_ylabel('PC2')
        axes[0,1].set_title(f'{mode.upper()}: PCA Pre vs Post-fire')
    axes[0,1].legend()
    
    # Plot 3: Time series PC1
    pre_fire_indices = [i for i, date in enumerate(sorted_dates_dt) if date < fire_date]
    post_fire_indices = [i for i, date in enumerate(sorted_dates_dt) if date >= fire_date]
    
    if pre_fire_indices:
        axes[1,0].plot(np.array(sorted_dates_dt)[pre_fire_indices], 
                     sorted_embeddings_pca[pre_fire_indices, 0], 'o-', 
                     color='lightgray', alpha=0.7, label='Pre-fire')
    if post_fire_indices:
        axes[1,0].plot(np.array(sorted_dates_dt)[post_fire_indices], 
                     sorted_embeddings_pca[post_fire_indices, 0], 'o-', 
                     color='red', alpha=0.7, label='Post-fire')
    
    axes[1,0].axvline(x=fire_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1,0].set_ylabel('PC1 of Embeddings')
    axes[1,0].set_title(f'{mode.upper()}: PC1 Time Series')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: NDVI time series
    if pre_fire_indices:
        axes[1,1].plot(np.array(sorted_dates_dt)[pre_fire_indices], 
                     sorted_ndvi[pre_fire_indices], 'o-', 
                     color='green', alpha=0.7, label='Pre-fire')
    if post_fire_indices:
        axes[1,1].plot(np.array(sorted_dates_dt)[post_fire_indices], 
                     sorted_ndvi[post_fire_indices], 'o-', 
                     color='darkred', alpha=0.7, label='Post-fire')
    
    axes[1,1].axvline(x=fire_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1,1].set_ylabel('NDVI')
    axes[1,1].set_title(f'{mode.upper()}: NDVI Time Series')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Plot 5: PCA explained variance
    axes[2,0].bar(range(1, len(pca.explained_variance_ratio_)+1), 
               pca.explained_variance_ratio_, alpha=0.7)
    axes[2,0].set_title(f'{mode.upper()}: PCA Explained Variance')
    axes[2,0].set_xlabel('Principal Component')
    axes[2,0].set_ylabel('Explained Variance Ratio')
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 6: NDVI vs PC1 correlation
    if len(ndvi_array) > 1:
        axes[2,1].scatter(ndvi_array, embeddings_pca[:, 0], 
                       c=[0 if c == 'lightgray' else 1 for c in colors],
                       cmap='RdGy_r', alpha=0.7, s=50)
        correlation = np.corrcoef(ndvi_array, embeddings_pca[:, 0])[0, 1]
        axes[2,1].set_title(f'{mode.upper()}: NDVI vs PC1 (r={correlation:.3f})')
        axes[2,1].set_xlabel('NDVI')
        axes[2,1].set_ylabel('First Principal Component')
        axes[2,1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(ndvi_array, embeddings_pca[:, 0], 1)
        p = np.poly1d(z)
        axes[2,1].plot(ndvi_array, p(ndvi_array), "r--", alpha=0.8)
    else:
        axes[2,1].text(0.5, 0.5, 'Insufficient data', 
                    ha='center', va='center', transform=axes[2,1].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{mode}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Seasonal analysis with residuals
    if len([x for x in pre_fire_mask if x]) > 4:
        try:
            pre_fire_days = np.array(day_of_year)[pre_fire_mask]
            pre_fire_pc1 = embeddings_pca[pre_fire_mask, 0]
            
            popt, _ = curve_fit(seasonal_curve, pre_fire_days, pre_fire_pc1)
            
            # Calculate residuals
            all_days = np.array(day_of_year)
            predicted = seasonal_curve(all_days, *popt)
            residuals = embeddings_pca[:, 0] - predicted
            
            # Plot seasonal fit and residuals
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f'{mode.upper()}: Seasonal Pattern Analysis', fontsize=14, fontweight='bold')
            
            # Seasonal fit plot
            for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
                mask = [c == color for c in colors]
                if any(mask):
                    ax1.scatter(np.array(day_of_year)[mask], embeddings_pca[mask, 0], 
                             c=color, alpha=0.7, s=50, label=label)
            
            x_smooth = np.linspace(1, 365, 365)
            y_smooth = seasonal_curve(x_smooth, *popt)
            ax1.plot(x_smooth, y_smooth, 'k-', linewidth=2, label='Seasonal fit')
            ax1.axvline(x=fire_day_of_year, color='red', linestyle='--', linewidth=2)
            ax1.set_xlabel('Day of Year')
            ax1.set_ylabel('First Principal Component')
            ax1.set_title('Seasonal Pattern with Fire Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            for color, label in zip(['lightgray', 'red'], ['Pre-fire', 'Post-fire']):
                mask = [c == color for c in colors]
                if any(mask):
                    ax2.scatter(np.array(day_of_year)[mask], np.abs(residuals)[mask], 
                             c=color, alpha=0.7, s=50, label=label)
            
            ax2.axvline(x=fire_day_of_year, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Day of Year')
            ax2.set_ylabel('Absolute Residual')
            ax2.set_title('Deviation from Seasonal Pattern')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'{mode}_seasonal_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Could not fit seasonal curve for {mode}: {e}")
    
    # Calculate and print statistics
    print(f"\n=== {mode.upper()} FIRE DETECTION ANALYSIS ===")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
    
    pre_fire_pc1 = embeddings_pca[pre_fire_mask, 0]
    post_fire_pc1 = embeddings_pca[[not x for x in pre_fire_mask], 0]
    pre_fire_ndvi = ndvi_array[pre_fire_mask] 
    post_fire_ndvi = ndvi_array[[not x for x in pre_fire_mask]]
    
    if len(pre_fire_pc1) > 0 and len(post_fire_pc1) > 0:
        print(f"Pre-fire PC1: {np.mean(pre_fire_pc1):.3f} ± {np.std(pre_fire_pc1):.3f}")
        print(f"Post-fire PC1: {np.mean(post_fire_pc1):.3f} ± {np.std(post_fire_pc1):.3f}")
        print(f"Pre-fire NDVI: {np.mean(pre_fire_ndvi):.3f} ± {np.std(pre_fire_ndvi):.3f}")
        print(f"Post-fire NDVI: {np.mean(post_fire_ndvi):.3f} ± {np.std(post_fire_ndvi):.3f}")
        
        # Statistical tests
        pc1_ttest = ttest_ind(pre_fire_pc1, post_fire_pc1)
        ndvi_ttest = ttest_ind(pre_fire_ndvi, post_fire_ndvi)
        print(f"PC1 t-test p-value: {pc1_ttest.pvalue:.4f}")
        print(f"NDVI t-test p-value: {ndvi_ttest.pvalue:.4f}")
    
    return {
        'embeddings': embeddings_array,
        'pca': embeddings_pca,
        'tsne': embeddings_tsne,
        'dates': dates_array,
        'ndvi': ndvi_array,
        'pca_obj': pca,
        'pre_fire_mask': pre_fire_mask
    }

def compare_modes(results_dict, save_dir, fire_date):
    """Compare results across different modes"""
    print("Creating mode comparison plots...")
    
    modes = list(results_dict.keys())
    n_modes = len(modes)
    
    # Compare PC1 over time for all modes
    fig, axes = plt.subplots(n_modes, 1, figsize=(16, 4*n_modes))
    if n_modes == 1:
        axes = [axes]
    
    fig.suptitle('Mode Comparison: PC1 Fire Detection', fontsize=16, fontweight='bold')
    
    for i, mode in enumerate(modes):
        data = results_dict[mode]
        
        # Sort for proper time series
        sort_indices = np.argsort(data['dates'])
        sorted_dates_dt = [pd.to_datetime(data['dates'][idx]) for idx in sort_indices]
        sorted_pc1 = data['pca'][sort_indices, 0]
        
        pre_fire_indices = [j for j, date in enumerate(sorted_dates_dt) if date < fire_date]
        post_fire_indices = [j for j, date in enumerate(sorted_dates_dt) if date >= fire_date]
        
        if pre_fire_indices:
            axes[i].plot(np.array(sorted_dates_dt)[pre_fire_indices], 
                       sorted_pc1[pre_fire_indices], 'o-', 
                       color='lightgray', alpha=0.7, label='Pre-fire')
        if post_fire_indices:
            axes[i].plot(np.array(sorted_dates_dt)[post_fire_indices], 
                       sorted_pc1[post_fire_indices], 'o-', 
                       color='red', alpha=0.7, label='Post-fire')
        
        axes[i].axvline(x=fire_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[i].set_ylabel('PC1')
        axes[i].set_title(f'{mode.upper()}: PC1 Time Series')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mode_comparison_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare variance explained
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_pos = np.arange(len(modes))
    var_explained = [results_dict[mode]['pca_obj'].explained_variance_ratio_[0] for mode in modes]
    
    bars = ax.bar(x_pos, var_explained, alpha=0.7, 
                 color=['lightblue', 'lightgreen', 'lightcoral'][:len(modes)])
    ax.set_xlabel('Mode')
    ax.set_ylabel('PC1 Variance Explained')
    ax.set_title('PC1 Variance Explained by Mode')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([mode.upper() for mode in modes])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mode_comparison_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison statistics
    print("\n=== MODE COMPARISON SUMMARY ===")
    for mode in modes:
        data = results_dict[mode]
        print(f"{mode.upper()}:")
        print(f"  Embedding dimension: {data['embeddings'].shape[1]}")
        print(f"  PC1 variance explained: {data['pca_obj'].explained_variance_ratio_[0]:.3f}")
        
        pre_fire_pc1 = data['pca'][data['pre_fire_mask'], 0]
        post_fire_pc1 = data['pca'][[not x for x in data['pre_fire_mask']], 0]
        
        if len(pre_fire_pc1) > 0 and len(post_fire_pc1) > 0:
            pc1_diff = np.mean(post_fire_pc1) - np.mean(pre_fire_pc1)
            print(f"  Pre-fire PC1 mean: {np.mean(pre_fire_pc1):.3f}")
            print(f"  Post-fire PC1 mean: {np.mean(post_fire_pc1):.3f}")
            print(f"  Fire-induced PC1 change: {pc1_diff:.3f}")
            
            pc1_ttest = ttest_ind(pre_fire_pc1, post_fire_pc1)
            print(f"  PC1 t-test p-value: {pc1_ttest.pvalue:.4f}")
        print()
