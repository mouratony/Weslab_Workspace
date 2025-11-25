import sys
sys.path.append('../') 
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from py_wake.wind_turbines import WindTurbines
from py_wake.deficit_models.no_wake import NoWakeDeficit
from windFarms_windTurbines import *
from py_wake import XYGrid
from cluster_analysis import geoJson_coordinates_data

# UTM to lat/lon conversion function
def utm_to_latlon(easting, northing, zone_number=32, northern=True):
    proj_utm = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84', northern=northern)
    lon, lat = proj_utm(easting, northing, inverse=True)
    return lat, lon

# Selective label formatter
def selective_label_formatter(tick_values_to_label, fmt="{:.2f}°"):
    def formatter(x, pos):
        if np.round(x, 4) in np.round(tick_values_to_label, 4):
            return fmt.format(x)
        else:
            return ''
    return ticker.FuncFormatter(formatter)

site = Rodsand_2()
ws = 10
wd = 90
windTurbines = WindTurbines.from_WindTurbine_lst([SWT_23_93(), SWT_23_82()])
windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]
wt=SWT_23_93()
deficitModels =  {
         'NOJ': noj_WF_model(site, windTurbines),
          'Bastankhah': bastankhah_WF_model(site, windTurbines),
          'TurboNoj': turboNoj_WF_model(site, windTurbines),
          'SuperGaussian': blondelSuperGaussian_WF_model(site, windTurbines),
          'Niayifar': niayifar_WF_model(site, windTurbines),
          'Zong': zong_WF_model(site, windTurbines),
          'Nygaard': nygaard_WF_model(site, windTurbines),
          'Carbajo': carbajo_WF_model(site, windTurbines),
        }

# Extract coordinates clearly
Ny_wt_x, Ny_wt_y = np.load("../wind_turbine_layouts/nysted_layout.npy")
Ro_wt_x, Ro_wt_y = np.load("../wind_turbine_layouts/rodsand2_layout.npy")

# Combine coordinates for NYRØ scenario
all_x = np.concatenate([Ro_wt_x, Ny_wt_x])
all_y = np.concatenate([Ro_wt_y, Ny_wt_y])
types = [0]*len(Ro_wt_x) + [1]*len(Ny_wt_x)

# Create a common XYGrid for baseline and other simulations
x_coords = np.linspace(min(all_x)-10000, max(all_x)+7000, 1000)
y_coords = np.linspace(min(all_y)-5000, max(all_y)+8000, 1000)

common_grid = XYGrid(x=x_coords, y=y_coords, h=69)


# Wake scenario NYRØ (Both farms)
wf_model_NYRO = noj_WF_model(site, windTurbines)
sim_NYRO = wf_model_NYRO(all_x, all_y, type=types, ws=ws, wd=wd)
flow_NYRO = sim_NYRO.flow_map(grid=common_grid)
WS_NYRO = flow_NYRO.WS_eff.squeeze()

# Baseline scenario RØ (Only Rødsand II wakes, NO Nysted turbines!)
sim_RO = wf_model_NYRO(Ro_wt_x, Ro_wt_y, type=0, ws=ws, wd=wd)
flow_RO = sim_RO.flow_map(grid=common_grid)
WS_RO = flow_RO.WS_eff.squeeze()

WS_diff_rel = ((WS_NYRO - WS_RO) / WS_RO * 100).values

X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

# Convert grid coordinates
lat_grid, lon_grid = utm_to_latlon(X_grid, Y_grid)
turbine_lat, turbine_lon = utm_to_latlon(all_x, all_y)

fig, axes = plt.subplots(4, 2, figsize=(20, 12))
axes = axes.flatten()
# Define clear color levels for all plots
levels = np.linspace(-15, 15, 31)

# Store a reference for colorbar normalization
contour_ref = None

# Axis label formatting (only specific labels shown)
lon_tick_labels = [11.4, 11.6]
lat_tick_labels = [54.52, 54.56, 54.60, 54.64]

for idx, (model_name, wfm) in enumerate(deficitModels.items()):
    ax = axes[idx]
    
    # NYRØ scenario (both wind farms)
    sim_NYRO = wfm(all_x, all_y, type=types, ws=ws, wd=wd)
    flow_NYRO = sim_NYRO.flow_map(grid=common_grid)
    WS_NYRO = flow_NYRO.WS_eff.squeeze()

    # RØ scenario (only Rødsand II wind farm)
    sim_RO = wfm(Ro_wt_x, Ro_wt_y, type=0, ws=ws, wd=wd)
    flow_RO = sim_RO.flow_map(grid=common_grid)
    WS_RO = flow_RO.WS_eff.squeeze()

    # Relative wind speed difference
    WS_diff_rel = ((WS_NYRO - WS_RO) / WS_RO * 100).values

    # Plot contour
    contour = ax.contourf(
        lon_grid, lat_grid, WS_diff_rel,
        levels=levels,
        cmap='RdBu',
        extend='both'
    )

    if contour_ref is None:
        contour_ref = contour

    # Scatter turbines
    ax.scatter(turbine_lon, turbine_lat, marker='o', s=8, facecolor='g', alpha=0.3, edgecolors='g')
    ax.grid(True)
    ax.set_aspect('auto')

    # Setting axis limits
    ax.set_xlim(11.3, 11.8)
    ax.set_ylim(54.50, 54.645)


    ax.xaxis.set_major_formatter(selective_label_formatter(lon_tick_labels, "{:.1f}°E"))
    ax.yaxis.set_major_formatter(selective_label_formatter(lat_tick_labels, "{:.2f}°N"))

    ax.set_title(model_name, fontsize=18)

# Remove unused subplot (since you have 15 models at most, adjust if fewer)
for i in range(len(deficitModels), len(axes)):
    fig.delaxes(axes[i])


# Adjust subplot positions and spacing
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.4, wspace=0.3)

# Explicitly position colorbar below the plots
cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])  # Adjust if needed
cbar = fig.colorbar(contour_ref, cax=cbar_ax, orientation='horizontal', ticks=np.arange(-15, 16, 5))
cbar.set_label(r'$(WS_{hub,NYRØ}-WS_{hub,RØ})/WS_{hub,RØ}$ [%]', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()
