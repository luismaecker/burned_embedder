import numpy as np
import rioxarray as rxr
import os
import glob
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask


def setup_dask_client(n_workers=2, threads_per_worker=2, memory_limit='4GB'):
    """Setup Dask client with controlled resources"""
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=':8787',
            silence_logs=False
        )
        
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}")
        print(f"Workers: {n_workers}, Threads per worker: {threads_per_worker}")
        print(f"Memory limit per worker: {memory_limit}")
        
        return client, cluster
    
    except Exception as e:
        print(f"Failed to create Dask cluster: {e}")
        return None, None

def get_tif_files(folder_path):
    """Get all .tif files in a folder"""
    return sorted(glob.glob(os.path.join(folder_path, "*.tif")) + 
                  glob.glob(os.path.join(folder_path, "*.tiff")))

def create_deforestation_mask_rioxarray(input_path, output_folder, confidence_level='high', chunk_size=(5000, 5000)):
    """Create binary deforestation mask using rioxarray"""
    
    tile_name = Path(input_path).stem
    print(f"Processing {tile_name} with rioxarray...")
    
    try:
        # Load raster with rioxarray (automatically uses Dask if chunked)
        print(f"[{tile_name}] Loading raster...")
        raster = rxr.open_rasterio(input_path, chunks=chunk_size)
        
        # Get the first band
        raster_data = raster.isel(band=0)
        
        print(f"[{tile_name}] Raster shape: {raster_data.shape}, chunks: {raster_data.chunks}")
        
        # Create masks
        print(f"[{tile_name}] Creating deforestation mask...")
        
        valid_mask = raster_data > 0
        first_digit = raster_data // 10000
        
        # Apply confidence level filter
        if confidence_level == 'high':
            deforestation_mask = (first_digit == 3) & valid_mask
        elif confidence_level == 'low':
            deforestation_mask = (first_digit == 2) & valid_mask
        else:  # all
            deforestation_mask = valid_mask
        
        # Convert to binary (1 = deforested, 0 = not deforested)
        binary_mask = deforestation_mask.astype('uint8')
        
        print(f"[{tile_name}] Computing statistics...")
        
        # Compute statistics
        total_pixels = binary_mask.size
        deforested_pixels = binary_mask.sum().compute().item()
        deforestation_rate = deforested_pixels / total_pixels * 100
        
        print(f"[{tile_name}] Deforested pixels: {deforested_pixels:,} ({deforestation_rate:.3f}%)")
        
        # Save binary mask
        output_path = os.path.join(output_folder, f"{tile_name}_mask.tif")
        
        print(f"[{tile_name}] Saving mask...")
        binary_mask.rio.to_raster(
            output_path,
            compress='lzw',
            dtype='uint8'
        )
        
        # Close the dataset
        raster.close()
        
        print(f"[{tile_name}] ✓ Saved mask: {os.path.basename(output_path)}")
        
        return {
            'tile_name': tile_name,
            'input_path': input_path,
            'output_path': output_path,
            'total_pixels': int(total_pixels),
            'deforested_pixels': int(deforested_pixels),
            'deforestation_rate': float(deforestation_rate),
            'success': True
        }
        
    except Exception as e:
        print(f"[{tile_name}] ✗ ERROR: {e}")
        return {
            'tile_name': tile_name,
            'input_path': input_path,
            'success': False,
            'error': str(e)
        }

def create_masks_for_folder(input_folder, output_folder, confidence_level='high', 
                           chunk_size=(5000, 5000), n_workers=2, threads_per_worker=2, 
                           memory_limit='4GB'):
    """Create deforestation masks for all tiles in folder using rioxarray with Dask client"""
    
    print(f"=== RIOXARRAY + DASK DEFORESTATION MASK CREATION ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Confidence level: {confidence_level}")
    print(f"Chunk size: {chunk_size}")
    
    # Setup Dask client
    client, cluster = setup_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    if not client:
        print("Warning: Failed to setup Dask client, using default scheduler")
    
    try:
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all TIF files
        tif_files = get_tif_files(input_folder)
        
        if not tif_files:
            print(f"No TIF files found in {input_folder}")
            return None
        
        print(f"Found {len(tif_files)} TIF files")
        print(f"Processing tiles sequentially with rioxarray + Dask...\n")
        
        # Process tiles one by one
        results = []
        
        for i, tif_file in enumerate(tif_files, 1):
            print(f"=== TILE {i}/{len(tif_files)} ===")
            
            result = create_deforestation_mask_rioxarray(
                input_path=tif_file,
                output_folder=output_folder,
                confidence_level=confidence_level,
                chunk_size=chunk_size
            )
            
            results.append(result)
            print()  # Empty line between tiles
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"=== SUMMARY ===")
        print(f"Successfully processed: {len(successful_results)}/{len(tif_files)} tiles")
        
        if failed_results:
            print(f"\nFailed tiles:")
            for result in failed_results:
                print(f"  {result['tile_name']}: {result['error']}")
        
        if successful_results:
            total_pixels = sum(r['total_pixels'] for r in successful_results)
            total_deforested = sum(r['deforested_pixels'] for r in successful_results)
            overall_rate = total_deforested / total_pixels * 100
            
            print(f"\nOverall statistics:")
            print(f"Total pixels: {total_pixels:,}")
            print(f"Deforested pixels: {total_deforested:,}")
            print(f"Overall deforestation rate: {overall_rate:.3f}%")
            
            print(f"\nPer-tile breakdown:")
            for result in successful_results:
                print(f"  {result['tile_name']}: {result['deforestation_rate']:.3f}% "
                      f"({result['deforested_pixels']:,} pixels)")
            
            print(f"\nMask files saved in: {output_folder}")
        
        return results
    
    finally:
        # Cleanup Dask resources
        if client:
            client.close()
        if cluster:
            cluster.close()

def create_mask_single_file(input_file, output_folder, confidence_level='high', 
                           chunk_size=(5000, 5000), n_workers=2, threads_per_worker=2, 
                           memory_limit='4GB'):
    """Create mask for a single file using rioxarray with Dask client"""
    
    print(f"=== SINGLE FILE RIOXARRAY + DASK MASK CREATION ===")
    print(f"Input file: {input_file}")
    print(f"Output folder: {output_folder}")
    print(f"Confidence level: {confidence_level}")
    print(f"Chunk size: {chunk_size}")
    
    # Setup Dask client
    client, cluster = setup_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    if not client:
        print("Warning: Failed to setup Dask client, using default scheduler")
    
    try:
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Process single file
        result = create_deforestation_mask_rioxarray(
            input_path=input_file,
            output_folder=output_folder,
            confidence_level=confidence_level,
            chunk_size=chunk_size
        )
        
        if result['success']:
            print(f"\n=== COMPLETED ===")
            print(f"Tile: {result['tile_name']}")
            print(f"Deforestation rate: {result['deforestation_rate']:.3f}%")
            print(f"Output: {result['output_path']}")
        else:
            print(f"Failed to process {result['tile_name']}: {result['error']}")
        
        return result
    
    finally:
        # Cleanup Dask resources
        if client:
            client.close()
        if cluster:
            cluster.close()

# Main execution
if __name__ == "__main__":
    
    # Configuration
    CONFIDENCE_LEVEL = 'high'  # 'high', 'low', or 'all'
    OUTPUT_FOLDER = "data/radd/masks"
    CHUNK_SIZE = (5000, 5000)  # Adjust based on your memory
    
    # Dask configuration
    N_WORKERS = 2  # Number of worker processes
    THREADS_PER_WORKER = 2  # Threads per worker
    MEMORY_LIMIT = '4GB'  # Memory limit per worker
    
    # Choose processing mode
    PROCESS_SINGLE_FILE = False  # Set to True for single file mode
    
    print(f"=== RIOXARRAY + DASK DEFORESTATION MASK GENERATOR ===")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Confidence level: {CONFIDENCE_LEVEL}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Dask workers: {N_WORKERS} (threads per worker: {THREADS_PER_WORKER})")
    print(f"Memory limit per worker: {MEMORY_LIMIT}")
    print(f"Output format: Binary TIF (1=deforested, 0=not deforested)")
    print(f"Dashboard: Will be available at http://localhost:8787\n")
    
    if PROCESS_SINGLE_FILE:
        # Single file mode
        INPUT_FILE = "data/raw/10N_080W.tif"
        
        result = create_mask_single_file(
            input_file=INPUT_FILE,
            output_folder=OUTPUT_FOLDER,
            confidence_level=CONFIDENCE_LEVEL,
            chunk_size=CHUNK_SIZE,
            n_workers=N_WORKERS,
            threads_per_worker=THREADS_PER_WORKER,
            memory_limit=MEMORY_LIMIT
        )
        
    else:
        # Folder mode
        INPUT_FOLDER = "data/raw/radd/south_america"
        
        results = create_masks_for_folder(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            confidence_level=CONFIDENCE_LEVEL,
            chunk_size=CHUNK_SIZE,
            n_workers=N_WORKERS,
            threads_per_worker=THREADS_PER_WORKER,
            memory_limit=MEMORY_LIMIT
        )
        
        if results:
            successful_count = sum(1 for r in results if r['success'])
            print(f"\nFinal result: {successful_count} masks created successfully")