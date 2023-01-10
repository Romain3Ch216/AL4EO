import numpy as np
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

src = rasterio.open(input_file)

with rasterio.Env():
    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw')
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(np.zeros((1, src.width, src.height)).astype(rasterio.uint8))
