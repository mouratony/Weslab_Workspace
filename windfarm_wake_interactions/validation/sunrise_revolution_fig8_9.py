import sys
sys.path.append('../') 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from py_wake.wind_turbines import WindTurbines
from windFarms_windTurbines import *
from py_wake import XYGrid, Points
from axis_formater import meter_to_km, scale_axis
from cluster_analysis import *
from layout_dev import _plot_wake
import math
import csv

with open('/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/data.csv', 'r') as f:
    data_array = np.loadtxt(f, delimiter=',')

with open('/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/les_overlayfig8.csv'
          , 'r') as f:
    _data_array = np.loadtxt(f, delimiter=',')

with open('/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/mesoscale_fig8.csv'
          , 'r') as f:
    mes_data_array = np.loadtxt(f, delimiter=',')
# Print CSV headers/columns
x_data = [float(x) for x in data_array[:,0]]
y_data = [float(y) for y in data_array[:,1]]

_LES_x_data = [float(x) for x in _data_array[:,0]]
_LES_y_data = [float(y) for y in _data_array[:,1]]
# print(x_data)
# print(y_data)
x_mes = [float(x) for x in mes_data_array[:,0]]
y_mes = [float(y) for y in mes_data_array[:,1]]



plt.figure()
plt.title('Figure 8, ws=8.83 m/s, wd=223°')
plt.plot(x_data, y_data, label='LES', lw=2)
plt.plot(_LES_x_data, _LES_y_data, color = 'b', lw=3, alpha=0.2)
plt.plot(x_mes, y_mes, label='Mesoscale', lw=2, color='b', ls='--')
plt.xlim(-22.5, 25)
plt.ylim(6.0, 8.9)
plt.legend()


# with open('/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/les_fig8.json', 'r') as f2:
#     LES_data_case_2 = json.load(f2)

# print(LES_data.keys())
# exit(0)
class scaled_turbine(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        hub height site specific
        """
        GenericWindTurbine.__init__(self, name='AD-5-116', diameter=206, hub_height=133,
                             power_norm=11000, turbulence_intensity=0.07)


turbine_locations = [
    (41.228, -71.085),
    (41.228, -71.063),
    (41.227, -71.129),
    (41.227, -71.107),
    (41.209, -71.194),
    (41.209, -71.172),
    (41.210, -71.106),
    (41.211, -71.084),
    (41.192, -71.216),
    (41.192, -71.194),
    (41.193, -71.128),
    (41.194, -71.106),
    (41.175, -71.215),
    (41.176, -71.171),
    (41.176, -71.149),
    (41.177, -71.127),
    (41.177, -71.105),
    (41.158, -71.259),
    (41.158, -71.237),
    (41.158, -71.215),
    (41.159, -71.171),
    (41.160, -71.149),
    (41.211, -71.062),
    (41.193, -71.150),
    (41.175, -71.193),
    (41.160, -71.127),
    (41.161, -71.082),
    (41.161, -71.060),
    (41.162, -71.038),
    (41.162, -71.016),
    (41.162, -70.994),
    (41.144, -71.082),
    (41.145, -71.038),
    (41.111, -71.059),
    (41.094, -71.058),
    (41.096, -70.970),
    (41.097, -70.949),
    (41.078, -71.058),
    (41.078, -71.036),
    (41.080, -70.948),
    (41.163, -70.973),
    (41.128, -71.081),
    (41.159, -71.191),
    (41.111, -71.038),
    (41.095, -71.035),
    (41.097, -70.992),
    (41.079, -71.015),
    (41.128, -71.037),
    (41.079, -70.990),
    (41.131, -70.840),
    (41.131, -70.905),
    (41.131, -70.883),
    (41.131, -70.861),
    (41.114, -70.905),
    (41.114, -70.882),
    (41.115, -70.860),
    (41.115, -70.839),
    (41.161, -71.105),
    (41.113, -70.927),
    (41.095, -71.014),
    (41.112, -70.993),
    (41.144, -71.126),
    (41.144, -71.103),
    (41.226, -71.173),
    (41.226, -71.151),
    (41.109, -71.190),
    (41.092, -71.168),
    (41.093, -71.123),
    (41.078, -71.102),
    (41.109, -71.125),
    (41.109, -71.169),
    (41.110, -71.147),
    (41.092, -71.191),
    (41.094, -71.102),
    (41.075, -71.190),
    (41.076, -71.146),
    (41.077, -71.124),
    (41.025, -71.188),
    (41.025, -71.210),
    (41.025, -71.232),
    (41.024, -71.254),
    (41.024, -71.276),
    (41.013, -70.924),
    (41.013, -70.946),
    (41.013, -70.968),
    (41.012, -70.990),
    (41.012, -71.012),
    (41.012, -71.034),
    (40.997, -70.901),
    (40.997, -70.923),
    (40.996, -70.945),
    (40.996, -70.989),
    (40.995, -71.011),
    (40.995, -71.033),
    (40.981, -70.901),
    (40.980, -70.923),
    (40.980, -70.945),
    (40.979, -70.967),
    (40.979, -70.989),
    (40.979, -71.011),
    (40.978, -71.033),
    (40.963, -70.922),
    (40.963, -70.944),
    (40.963, -70.966),
    (40.962, -70.988),
    (40.946, -70.944),
    (40.946, -70.966),
    (40.946, -70.988),
    (40.945, -71.010),
    (40.945, -71.032),
    (40.930, -70.943),
    (40.929, -70.965),
    (40.929, -70.987),
    (40.929, -71.009),
    (40.928, -71.031),
    (40.912, -70.987),
    (40.912, -71.009),
    (40.912, -71.031),
    (41.046, -70.947),
    (41.011, -71.078),
    (41.010, -71.144),
    (41.009, -71.166),
    (41.009, -71.188),
    (41.008, -71.210),
    (41.008, -71.254),
    (40.994, -71.099),
    (40.993, -71.143),
    (40.993, -71.165),
    (40.992, -71.187),
    (40.992, -71.209),
    (40.990, -71.275),
    (40.978, -71.055),
    (40.977, -71.077),
    (40.977, -71.099),
    (40.977, -71.121),
    (40.976, -71.143),
    (40.976, -71.165),
    (40.975, -71.187),
    (40.975, -71.209),
    (40.975, -71.231),
    (40.974, -71.253),
    (40.974, -71.275),
    (40.961, -71.054),
    (40.961, -71.076),
    (40.960, -71.098),
    (40.960, -71.142),
    (40.959, -71.164),
    (40.959, -71.186),
    (40.958, -71.230),
    (40.957, -71.252),
    (40.957, -71.274),
    (41.010, -71.122),
    (40.961, -71.032),
    (41.030, -70.903),
    (41.030, -70.946),
    (41.011, -71.100),
    (40.998, -70.879),
    (40.996, -70.967),
    (40.995, -71.054),
    (40.962, -71.011),
    (40.959, -71.120)
]



utm_val = [list(convert_LatLong_to_utm(lon, lat)) for lat, lon in turbine_locations]
wt_x_m, wt_y_m = zip(*utm_val)
wt_y, wt_x = zip(*turbine_locations)
# shs_wt_x, shs_wt_y = np.load("../wind_turbine_layouts/sheringham_shoal_layout.npy")

# plt.scatter(wt_x_m, wt_y_m)
# plt.tight_layout()
# plt.show()
grid = XYGrid(h=133, resolution=700, extend=1)
site = Revolutionwind_southforkwind()
windTurbines = scaled_turbine()
wd_mean = 205

#Case 3: ws=7.24 m/s, wd=219
ws = 8.83  #m/s
wd = 223
ws_min = 3.0 #m/s
#Case 4: ws=8.83 m/s, wd=208
# ws = 8.83  #m/s
# wd = 208
# ws_min = 3.0 #m/s

deficitModels_1 =  {
         'NOJ': [noj_WF_model(site, windTurbines), '#c65102', '--'],
          'Bastankhah': [bastankhah_WF_model(site, windTurbines), '#ff028d', '--'],
          'TurboNoj': [turboNoj_WF_model(site, windTurbines), '#a8ff04', '--'],
        # 'Nygaard-Original': [nygaard_original_model(site, windTurbines), 'r', '-'],
        # 'Nygaard-Revised' :[nygaard_paul_revised_model(site, windTurbines), 'r', '--']
        }

deficitModels_2 = {
        'Carbajo': [carbajo_WF_model(site, windTurbines), '#653700', '--'],
        'Niayifar': [niayifar_WF_model(site, windTurbines), '#8cffdb', '--'],
        'Zong': [zong_WF_model(site, windTurbines), '#fffd01','--'],
}

deficitModels_3 = {
    'Nygaard': [nygaard_WF_model(site, windTurbines), '#ff000d', '--'],
    'SuperGaussian': [blondelSuperGaussian_WF_model(site, windTurbines), "#900aa8", '--']
}

wfm = bastankhah_WF_model(site, windTurbines)

simres = wfm(wt_x_m, wt_y_m, h=133, n_cpu=None, ws=ws, wd=wd, time=True)

flow_map = simres.flow_map(grid=grid, wd=wd, ws=ws)

# plt.figure()
# # get contour set without auto–colorbar
# cs = flow_map.plot_wake_map(levels=500, cmap='viridis',
#                             plot_colorbar=False, plot_windturbines=True)
# # add your own colorbar with extend markers
# cb = plt.colorbar(cs, extend='both')
# cb.set_label('wind speed (m/s)')
# plt.show()

alpha_down = np.deg2rad((wd_mean + 180) % 360) 
# perpendicular cross-section orientation
alpha_cross = np.deg2rad((wd_mean - 90) % 360) 

# central reference point: mean position of all turbines
x0 = wt_x_m[30]
y0 = wt_y_m[30]

# offset 7.4 km downstream from the farm center
x_center = x0 + 7400.0 * np.sin(alpha_down)
y_center = y0 + 7400.0 * np.cos(alpha_down)

# distances along the transect from -25 km to +25 km
s = np.linspace(-50000.0, 50000.0, 225)

# compute x,y coordinates of each transect point
x_transect = x_center + s * np.sin(alpha_cross)
y_transect = y_center + s * np.cos(alpha_cross)
h_transect = np.full_like(x_transect, 133.0)  # constant hub height

fig, ax = plt.subplots()
ax.set_title(f'Simulation Case: Ū={ws} m/s, Θ={wd}°')

# our own contourf + colorbar
cs, cbar = _plot_wake(flow_map, ax=ax, levels=200, vmin=ws_min, vmax=ws, cmap='viridis', 
                      cbar_height= 5,

                       cbar_ticks=[4,6, 8] #case 3 
                       # cbar_ticks=[6, 8] #case 4 
                      )

# overlay transect and midpoint
ax.plot(x_transect, y_transect, 'k.', markersize=1.7)
ax.scatter([x_center], [y_center], color='r', zorder=5)

# axes labels/limits
xmin = min(wt_x_m); xmax = max(wt_x_m)
ymin = min(wt_y_m); ymax = max(wt_y_m)
ax.set_xlabel("X - UTM Coord [m]")
ax.set_ylabel("Y - UTM Coord [m]")
ax.set_xlim(xmin - 5000, xmax + 8500)
ax.set_ylim(ymin - 5000, ymax + 6000)
plt.tight_layout()

# create a Points grid for the transect
transect_grid = Points(x_transect, y_transect, h_transect)

# evaluate the flow field at the transect locations
transect_map = simres.flow_map(grid=transect_grid, wd=wd, ws=ws)
transect_ws = transect_map.WS_eff.squeeze()



# plt.figure()
# plt.plot(s/1000, transect_ws)
# plt.xlim(-25, 27.5)
# plt.xlabel('Transect distance (km)')
# plt.ylabel('WS_eff (m/s)')
# plt.show()

fig, axes = plt.subplots(3, 1, sharex=True)
# Case 1
axes[0].plot(x_data, y_data, label='LES', lw=2)
axes[0].plot(_LES_x_data, _LES_y_data, color = 'b', lw=3, alpha=0.2)
axes[0].plot(x_mes, y_mes, color = 'blue', lw=2,  ls='--', label='Mesoscale')
axes[1].plot(x_data, y_data, lw=2)
axes[1].plot(_LES_x_data, _LES_y_data, color = 'b', lw=3, alpha=0.2)
axes[1].plot(x_mes, y_mes, color = 'blue', lw=2,  ls='--')
axes[2].plot(x_data, y_data, lw=2)
axes[2].plot(_LES_x_data, _LES_y_data, color = 'b', lw=3, alpha=0.2)
axes[2].plot(x_mes, y_mes, color = 'blue', lw=2,  ls='--')



# Case 2
# axes[0].plot(LES_data_case_2['distance'], LES_data_case_2['WindSpeed'], color='orange', label='LES', lw=2, alpha=0.8)
# axes[1].plot(LES_data_case_2['distance'], LES_data_case_2['WindSpeed'], color='orange', lw=2, alpha=0.8)
# axes[2].plot(LES_data_case_2['distance'], LES_data_case_2['WindSpeed'], color='orange', lw=2, alpha=0.8)

# Add all models from the dictionary to the same figure
for name, (model, color, ls) in deficitModels_1.items():
    simres_i = model(wt_x_m, wt_y_m, h=133, n_cpu=None, ws=ws, wd=wd, time=True)
    transect_map_i = simres_i.flow_map(grid=transect_grid, wd=wd, ws=ws)
    transect_ws_i = transect_map_i.WS_eff.squeeze()
    axes[0].plot(s/1000, transect_ws_i, color=color, linestyle=ls, label=name)
    axes[0].plot([0, 0], [0, ws], 'k', ls='--', lw=1)
    axes[0].set_ylim(6.0, 8.9)
    axes[0].set_xlim(-22.5, 27.5)
    axes[0].set_ylabel('Ū[m/s]')

for name, (model, color, ls) in deficitModels_2.items():
    simres_i = model(wt_x_m, wt_y_m, h=133, n_cpu=None, ws=ws, wd=wd, time=True)
    transect_map_i = simres_i.flow_map(grid=transect_grid, wd=wd, ws=ws)
    transect_ws_i = transect_map_i.WS_eff.squeeze()
    axes[1].plot(s/1000, transect_ws_i, color=color, linestyle=ls, label=name)
    axes[1].plot([0, 0], [0, ws], 'k', ls='--', lw=1)
    axes[1].set_ylim(6.0, 8.9)
    axes[1].set_xlim(-22.5, 27.5)
    axes[1].set_ylabel('Ū[m/s]')

for name, (model, color, ls) in deficitModels_3.items():
    simres_i = model(wt_x_m, wt_y_m, h=133, n_cpu=None, ws=ws, wd=wd, time=True)
    transect_map_i = simres_i.flow_map(grid=transect_grid, wd=wd, ws=ws)
    transect_ws_i = transect_map_i.WS_eff.squeeze()
    axes[2].plot([0, 0], [0, ws], 'k', ls='--', lw=1,)
    axes[2].plot(s/1000, transect_ws_i, color=color, linestyle=ls, label=name)
    axes[2].set_ylabel('Ū[m/s]')
    axes[2].set_ylim(6.0, 8.9)

handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

fig.legend(handles, labels,
           loc='upper center',
           ncol=6,           # adjust as needed
           bbox_to_anchor=(0.5, 1), fontsize=8, frameon=False)

plt.xlim(-22.5, 25)
plt.xlabel('Transect distance (km)')
plt.tight_layout()  # leave space for legend
plt.show()
