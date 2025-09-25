# TerraFM Setup Instructions

This document provides instructions for setting up TerraFM to replace Copernicus-FM in your fire detection analysis.

## Step 1: Download TerraFM Files

### 1.1 Download Model Weights
Visit the TerraFM model zoo and download the pretrained weights:
- **TerraFM-B.pth** (Base model, recommended for most use cases)
- **TerraFM-L.pth** (Large model, for better performance but higher memory usage)

Store the downloaded weights file in a accessible location, e.g.:
```
/path/to/your/models/TerraFM-B.pth
```

### 1.2 Download terrafm.py
Download the `terrafm.py` file from the TerraFM repository and place it in a directory accessible to your Python environment.

## Step 2: Update Configuration

### 2.1 Set TerraFM Path
In your script (`duebener_fire.py`), uncomment and update the TerraFM path:

```python
import sys
sys.path.append('/path/to/terrafm')  # Update this path
```

### 2.2 Set Model Weights Path
In the `main()` function, update the weights path:

```python
weights_path = "/path/to/your/models/TerraFM-B.pth"  # Update this path
```

## Step 3: Key Changes Made

### 3.1 Model Loading
- Replaced Copernicus-FM with TerraFM
- Added support for both base and large model sizes
- Added weight loading functionality

### 3.2 Data Processing
- **All modes now use 15 channels**: TerraFM expects all 15 channels (2 S1 + 13 S2)
- **Zero padding**: Missing modalities are padded with zeros:
  - `s1_only`: S1 data + zero-padded S2 channels
  - `s2_only`: Zero-padded S1 channels + S2 data
  - `combined`: S1 + S2 data (no padding needed)

### 3.3 Forward Pass
- **Simplified interface**: No metadata, wavelengths, or bandwidths needed
- **Direct model call**: `encoder(x)` instead of complex parameter passing

## Step 4: Benefits of TerraFM

1. **Joint Training**: Trained on both S1 and S2 data together
2. **Better Representations**: May provide better embeddings for fire detection
3. **Simpler Interface**: Less complex parameter management
4. **Consistent Input**: Always uses all 15 channels for consistency

## Step 5: Testing

After setup, test your configuration:

```bash
cd /path/to/burned_embedder
python scripts/duebener_fire.py
```

The script should now load TerraFM and process your data using the new model.

## Troubleshooting

### Import Errors
If you get import errors for `terrafm`:
1. Ensure `terrafm.py` is in your Python path
2. Check that the path in `sys.path.append()` is correct

### Memory Issues
If you encounter GPU memory issues:
- Try using the base model instead of large
- Reduce batch sizes in the processing function
- Ensure GPU memory is properly cleared between runs

### Weight Loading Issues
If weight loading fails:
- Verify the weights file path is correct
- Ensure the weights file is compatible with your model size choice
- Check that the weights file is not corrupted