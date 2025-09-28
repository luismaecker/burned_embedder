import rasterio
from datetime import datetime, timedelta
import rasterio.windows


def days_to_date(days_since_dec31_2014):
    """Convert days since Dec 31, 2014 to actual date"""
    base_date = datetime(2014, 12, 31)
    return base_date + timedelta(days=int(days_since_dec31_2014))


def load_sample_data(input_path, window):
    """Load raster data for a specific window"""
    with rasterio.open(input_path) as src:
        sample_data = src.read(1, window=window)
        raster_transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        window_transform = rasterio.windows.transform(window, raster_transform)
    
    return sample_data, window_transform, crs, profile


def decode_radd_alerts(sample_data, confidence_level):
    """Decode RADD alert data based on confidence level"""
    valid_mask = sample_data > 0
    first_digit = sample_data // 10000
    
    if confidence_level == 'high':
        analysis_mask = (first_digit == 3) & valid_mask
    elif confidence_level == 'low':
        analysis_mask = (first_digit == 2) & valid_mask
    else:
        analysis_mask = valid_mask
    
    return valid_mask, analysis_mask