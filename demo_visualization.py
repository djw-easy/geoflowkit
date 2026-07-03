"""Demo script for geoflowkit OD Matrix and MapTrix visualizations.

Uses Shenzhen taxi flow data (sz_taxi_flow.csv) and district boundaries
(sz_border.gpkg) to demonstrate both visualization methods.

Usage:  python demo_visualization.py
Output:
    demo_od_matrix.png         — basic heatmap
    demo_od_matrix_bubble.png  — heatmap + proportional circles (size_weight)
    demo_maptrix.png           — MapTrix layout with size circles
"""

import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

from geoflowkit import read_csv
from geoflowkit.visualization.od_matrix import ODMatrixVisualizer
from geoflowkit.visualization.maptrix import MapTrixVisualizer

# ---------------------------------------------------------------------------
# Global matplotlib style
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

fdf = read_csv(
    f'{DATA_DIR}/sz_taxi_flow.csv',
    use_cols=['ox', 'oy', 'dx', 'dy'],
    crs='EPSG:4326',
)
print(f'Loaded {len(fdf):,} flows')

N = 5000
np.random.seed(42)
indices = np.random.choice(len(fdf), size=N, replace=False)
fdf_sample = fdf.iloc[indices].reset_index(drop=True)
print(f'Using {len(fdf_sample):,} flows for visualization')

border = gpd.read_file(f'{DATA_DIR}/sz_border.gpkg')
print(f'Loaded {len(border)} districts: {border["Name"].tolist()}')

shenzhen_union = border.union_all()
fdf_sample = fdf_sample.clip(shenzhen_union)
print(f'After clipping to border: {len(fdf_sample)} flows remain')

# ---------------------------------------------------------------------------
# 2. OD Matrix — basic heatmap (color = flow count)
# ---------------------------------------------------------------------------

print('\n--- OD Matrix heatmap ---')

fig1, ax1 = plt.subplots(figsize=(10, 8))
ODMatrixVisualizer(
    origin_zones=border,
    zone_id_col='Name',
    weight='count',
    cmap='OrRd',
    show_labels=True,
    label_fontsize=9,
).fit_plot(fdf_sample, ax=ax1, colorbar=True)
ax1.set_title(
    'OD Matrix — Shenzhen Taxi Flows (by district, colour = count)',
    fontsize=14, weight='bold', pad=12,
)
fig1.tight_layout()
fig1.savefig('demo_od_matrix.png', dpi=150, bbox_inches='tight')
print('Saved demo_od_matrix.png')

# ---------------------------------------------------------------------------
# 3. OD Matrix — bubble matrix (colour = count, circle size = total length)
# ---------------------------------------------------------------------------

print('\n--- OD Matrix bubble (weight + size_weight) ---')

fig2, ax2 = plt.subplots(figsize=(10, 8))
ODMatrixVisualizer(
    origin_zones=border,
    zone_id_col='Name',
    weight='count',
    size_weight='length',
    cmap='OrRd',
    show_labels=True,
    label_fontsize=9,
).fit_plot(fdf_sample, ax=ax2, colorbar=True)
ax2.set_title(
    'OD Matrix Bubble — colour=count, circle size=flow length',
    fontsize=14, weight='bold', pad=12,
)
fig2.tight_layout()
fig2.savefig('demo_od_matrix_bubble.png', dpi=150, bbox_inches='tight')
print('Saved demo_od_matrix_bubble.png')

# ---------------------------------------------------------------------------
# 4. MapTrix — rotated matrix + maps + size circles
# ---------------------------------------------------------------------------

print('\n--- MapTrix figure ---')

fig3 = MapTrixVisualizer(
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
).fit_plot(fdf_sample, figsize=(16, 9))
fig3.savefig('demo_maptrix.png', dpi=150, bbox_inches='tight')
print('Saved demo_maptrix.png')

print('\nDone! Generated:')
print('  demo_od_matrix.png         — basic OD heatmap')
print('  demo_od_matrix_bubble.png  — OD heatmap + proportional circles')
print('  demo_maptrix.png           — MapTrix layout + size circles')
