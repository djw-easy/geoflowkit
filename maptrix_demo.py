"""Improved static MapTrix demo using Shenzhen taxi flow data.

This script demonstrates how to:
1. Load flow data from CSV and border polygons from GeoPackage
2. Sample flows and clip to the city boundary
3. Create a centred MapTrix matrix with non-crossing row/column leaders
4. Inspect the exported static layout geometry

Usage:
    python maptrix_demo.py
"""

import warnings

import numpy as np
import geopandas as gpd

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
visualizer = MapTrixVisualizer(
    origin_zones=border,
    zone_id_col='Name',
    weight='count',
    size_weight='length',
    matrix_cmap='YlOrRd',
    map_cmap='YlOrRd',
    origin_line_color='#2878B5',
    destination_line_color='#D97706',
    line_alpha=0.72,
    leader_linewidth=1.25,
    show_labels=True,
    label_fontsize=8,
    out_title='Origins · district outflow',
    in_title='Destinations · district inflow',
    matrix_title='Shenzhen taxi flows · origins × destinations',
    title_fontsize=12,
    include_self_flows=False,
)
fig = visualizer.fit_plot(fdf_sample, figsize=(16, 9))

fig.savefig('maptrix_demo.png', dpi=160, facecolor='white')
layout = visualizer.layout_
print(
    f'  Layout: {len(layout["origin_leaders"])} row leaders, '
    f'{len(layout["destination_leaders"])} column leaders'
)
print(f'  Shared row/column order: {layout["same_entity_set"]}')
print('Saved: maptrix_demo.png')
