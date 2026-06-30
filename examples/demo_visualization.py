"""Demo script for geoflowkit OD Matrix and MapTrix visualizations.

Uses Shenzhen taxi flow data (sz_taxi_flow.csv) and district boundaries
(sz_border.gpkg) to demonstrate both visualization methods.

Usage:  python demo_visualization.py
Output: demo_od_matrix.png, demo_maptrix.png
"""

import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from geoflowkit import read_csv, od_matrix, maptrix

# ---------------------------------------------------------------------------
# Global matplotlib style (matching project notebook conventions)
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

warnings.filterwarnings(
    'ignore', message='Geometry is in a geographic CRS',
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

DATA_DIR = 'examples/data/sz_data'

# Shenzhen taxi flow data
fdf = read_csv(
    f'{DATA_DIR}/sz_taxi_flow.csv',
    use_cols=['ox', 'oy', 'dx', 'dy'],
    crs='EPSG:4326',
)
print(f'Loaded {len(fdf):,} flows')

# Subsample for faster rendering (adjust N to taste)
N = 5000
np.random.seed(42)
indices = np.random.choice(len(fdf), size=N, replace=False)
fdf_sample = fdf.iloc[indices].reset_index(drop=True)
print(f'Using {len(fdf_sample):,} flows for visualization')

# Shenzhen district boundaries — each row is a zone polygon
border = gpd.read_file(f'{DATA_DIR}/sz_border.gpkg')
print(f'Loaded {len(border)} districts: {border["Name"].tolist()}')

# Clip flows to the border so all points fall inside zone polygons
shenzhen_union = border.union_all()
fdf_sample = fdf_sample.clip(shenzhen_union)
print(f'After clipping to border: {len(fdf_sample)} flows remain')

# ---------------------------------------------------------------------------
# 2. OD Matrix heatmap
# ---------------------------------------------------------------------------

print('\n--- Generating OD Matrix heatmap ---')

fig1, ax1 = plt.subplots(figsize=(10, 8))
od_matrix(
    fdf_sample,
    zones=border,               # each row = one zone
    zone_id_col='Name',         # use district name as zone label
    weight='count',
    cmap='OrRd',
    ax=ax1,
    colorbar=True,
    show_labels=True,
    label_fontsize=9,
)
ax1.set_title(
    'OD Matrix — Shenzhen Taxi Flows (by district)',
    fontsize=14, weight='bold', pad=12,
)
fig1.tight_layout()
fig1.savefig('demo_od_matrix.png', dpi=150, bbox_inches='tight')
print('Saved demo_od_matrix.png')

# ---------------------------------------------------------------------------
# 3. MapTrix figure
# ---------------------------------------------------------------------------

print('--- Generating MapTrix figure ---')

fig2 = maptrix(
    fdf_sample,
    zones=border,               # each row = one zone
    zone_id_col='Name',         # use district name as zone label
    weight='count',
    matrix_cmap='OrRd',
    line_color='black',
    line_alpha=0.5,
    show_labels=True,
    label_fontsize=9,
    out_title='Outflow',
    in_title='Inflow',
    title_fontsize=16,
    figsize=(16, 9),
    leader_center_weight=1.0,
    leader_sep_weight=8.0,
)
fig2.savefig('demo_maptrix.png', dpi=150, bbox_inches='tight')
print('Saved demo_maptrix.png')

print('\nDone! Open demo_od_matrix.png and demo_maptrix.png to view results.')
