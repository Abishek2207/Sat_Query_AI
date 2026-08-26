import pytest
from PIL import Image
import io
import rasterio
from rasterio.transform import from_origin
import numpy as np

@pytest.fixture
def valid_png_bytes():
    img = Image.new('RGB', (10, 10), color='blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@pytest.fixture
def synthetic_geotiff_bytes():
    transform = from_origin(0, 0, 1, 1)
    data = np.zeros((1, 10, 10), dtype=rasterio.uint8)
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w', driver='GTiff', height=10, width=10,
        count=1, dtype=data.dtype, crs='EPSG:4326', transform=transform
    ) as dataset:
        dataset.write(data)
    return mem_file.getvalue()
