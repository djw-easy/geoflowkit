"""Improved static MapTrix demo using Shenzhen taxi flow data.

This script demonstrates how to:
1. Load flow data from CSV and border polygons from GeoPackage
2. Sample flows and clip to the city boundary
3. Create a centred MapTrix matrix with diagonal-horizontal leaders
4. Adjust map size, map-to-matrix gap, and colorbar geometry
5. Encode regional totals with bounded leader/marker sizes
6. Inspect the exported static layout geometry

Usage:
    python examples/maptrix_demo.py
"""

import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd

from geoflowkit import read_csv
from geoflowkit.visualization import MapTrixVisualizer

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / 'data' / 'sz_data'
OUTPUT_PATH = EXAMPLE_DIR / 'maptrix_demo.png'

warnings.filterwarnings(
    'ignore', message='Geometry is in a geographic CRS',
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('Loading data ...')
fdf = read_csv(
    DATA_DIR / 'sz_taxi_flow.csv',
    use_cols=['ox', 'oy', 'dx', 'dy'],
    crs='EPSG:4326',
)
print(f'  {len(fdf):,} flows loaded')

border = gpd.read_file(DATA_DIR / 'sz_border.gpkg')
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
    map_cmap='viridis',
    origin_line_color='#2878B5',
    destination_line_color='#D97706',
    line_alpha=0.72,
    leader_routing='diagonal-horizontal',
    leader_angle=45,
    leader_width_range=(0.9, 5.0),
    centroid_size_range=(45, 340),
    origin_map_rect=(0.04, 0.56, 0.385, 0.36),
    destination_map_rect=(0.04, 0.08, 0.385, 0.36),
    matrix_rect=(0.346, 0.08, 0.525, 0.84),
    colorbar_rect=(0.85, 0.13, 0.014, 0.74),
    show_labels=True,
    label_fontsize=8,
    out_title='Origins · district outflow',
    in_title='Destinations · district inflow',
    title_fontsize=12,
    map_title_pad=10,
    include_self_flows=False,
)
fig = visualizer.fit_plot(fdf_sample, figsize=(16, 9))

fig.savefig(OUTPUT_PATH, dpi=160, facecolor='white')
layout = visualizer.layout_
print(
    f'  Layout: {len(layout["origin_leaders"])} row leaders, '
    f'{len(layout["destination_leaders"])} column leaders'
)
print(f'  Shared row/column order: {layout["same_entity_set"]}')
print(f'  Row order: {layout["row_order"]}')
print(f'  Column order: {layout["column_order"]}')
print(
    '  Minimum diagonal gaps (px): '
    f'{layout["minimum_diagonal_gap"]}'
)
print(f'Saved: {OUTPUT_PATH}')
