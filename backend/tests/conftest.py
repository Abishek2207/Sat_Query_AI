import pytest
from PIL import Image
import io
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

@pytest.fixture
def valid_png_bytes():
    img = Image.new('RGB', (10, 10), color='blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@pytest.fixture
def synthetic_geotiff_bytes():
    if not RASTERIO_AVAILABLE:
        # Return a small valid TIFF-like header just to bypass some basic checks if any, or just fake bytes
        # For routing, fake bytes might trigger validation failure. We will handle validation failure separately.
        return b"fake_tiff_bytes"
        
    transform = from_origin(0, 0, 1, 1)
    data = np.zeros((1, 10, 10), dtype=rasterio.uint8)
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w', driver='GTiff', height=10, width=10,
        count=1, dtype=data.dtype, crs='EPSG:4326', transform=transform
    ) as dataset:
        dataset.write(data)
    return mem_file.getvalue()
