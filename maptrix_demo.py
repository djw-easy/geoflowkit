"""MapTrix visualization demo using Shenzhen taxi flow data.

This script demonstrates how to:
1. Load flow data from CSV and border polygons from GeoPackage
2. Sample flows and clip to the city boundary
3. Create a MapTrix visualization with OD matrix + maps + guide lines

Usage:
    python examples/maptrix_demo.py
"""

import warnings

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from geoflowkit import read_csv
from geoflowkit.visualization import MapTrixVisualizer

DATA_DIR = 'examples/data/sz_data'

warnings.filterwarnings(
    'ignore', message='Geometry is in a geographic CRS',
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('Loading data ...')
fdf = read_csv(
    f'{DATA_DIR}/sz_taxi_flow.csv',
    use_cols=['ox', 'oy', 'dx', 'dy'],
    crs='EPSG:4326',
)
print(f'  {len(fdf):,} flows loaded')

border = gpd.read_file(f'{DATA_DIR}/sz_border.gpkg')
print(f'  {len(border)} districts: {border["Name"].tolist()}')

# ---------------------------------------------------------------------------
# 2. Sample & clip
# ---------------------------------------------------------------------------
N = 5000
np.random.seed(42)
indices = np.random.choice(len(fdf), size=N, replace=False)
fdf_sample = fdf.iloc[indices].reset_index(drop=True)
print(f'  Sampled {len(fdf_sample):,} flows')

shenzhen_union = border.union_all()
fdf_sample = fdf_sample.clip(shenzhen_union)
print(f'  After clipping: {len(fdf_sample):,} flows remain')

# ---------------------------------------------------------------------------
# 3. MapTrix visualization
# ---------------------------------------------------------------------------
print('Rendering MapTrix ...')
fig = MapTrixVisualizer(
    origin_zones=border,
    zone_id_col='Name',
    weight='count',
    size_weight='length',
    matrix_cmap='OrRd',
    line_color='black',
    line_alpha=0.5,
    show_labels=True,
    label_fontsize=9,
    out_title='Outflow',
    in_title='Inflow',
    title_fontsize=16,
    include_self_flows=False,
    width_ratios=[1, 1.5],
).fit_plot(fdf_sample, figsize=(16, 9))

fig.savefig('maptrix_demo.png', dpi=150, bbox_inches='tight')
print('Saved: maptrix_demo.png')
