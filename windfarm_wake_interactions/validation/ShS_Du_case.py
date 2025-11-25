import sys
sys.path.append('../') 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from py_wake.wind_turbines import WindTurbines
from windFarms_windTurbines import *
from py_wake import XYGrid
from axis_formater import meter_to_km, scale_axis
import math

SCADA_DATA = {'upper':[1.0475, 1.015, 1.005, 1.005, 0.9975, 0.99, 0.997, 0.979, 0.9925, 1.015, 1.03, 1.055], 
              'lower': [1.025, 0.995, 0.9825, 0.985, 0.9775, 0.97, 0.9775, 0.959, 0.97, 0.995, 1.01, 1.035]
              }

WRF_DATA = {'upper':[1.01, 1.004, 0.995, 0.986, 0.973, 0.96, 0.948, 0.937, 0.935, 0.9385, 0.947, 0.955, 
                     0.968, 0.9815, 0.995],
            'lower':[0.978, 0.97, 0.96, 0.95, 0.938, 0.925, 0.91, 0.9, 0.898, 0.903, 0.914, 0.925, 0.9335, 
                     0.95, 0.965]
}

RANS_AD_GA = {'wake_magnitude':[0.97, 0.96, 0.948, 0.94, 0.935, 0.932, 0.933, 0.934, 0.938, 0.944, 0.95, 0.955],
              'wake_shape':[1.025, 1.016, 1.005, 0.995, 0.99, 0.988, 0.989, 0.99, 0.994, 1.0, 1.006, 1.013], 
              'wake_mag_up':[0.994, 0.985, 0.971, 0.958, 0.9415, 0.93, 0.919, 0.913, 0.915, 0.92, 0.93, 0.94, 0.955, 0.97, 0.988]    
              }

RANS_AD_AWF = {'wake_magnitude':[0.974, 0.964, 0.955, 0.947, 0.941, 0.938, 0.9385, 0.9395, 0.9426, 0.948, 0.955, 0.96],
              'wake_shape':[1.023, 1.014, 1.004, 0.996, 0.991, 0.989, 0.9895, 0.99, 0.994, 0.998, 1.005, 1.011],     
              } 

RANS_AWF = [0.995, 0.985, 0.971, 0.9575, 0.942, 0.933, 0.925, 0.92, 0.922, 0.925, 0.935, 0.943, 0.956, 0.97, 0.983]

# Implementing gaussian filtering
theta_bar = 235
sigma_theta = 6
num_samples = 13  # odd number, centered at 235
theta_range = np.linspace(theta_bar - 3*sigma_theta,
                          theta_bar + 3*sigma_theta, num_samples)

weights = norm.pdf(theta_range, loc=theta_bar, scale=sigma_theta)
weights /= np.sum(weights)  # Normalize

def gaussian_filtered_ws_at_transect(wf_model, all_x, all_y, type, ws, transect_grid, theta_range, weights):
    weighted_ws_sum = np.zeros(len(transect_grid.x))  # shape (15,)
    
    for theta, weight in zip(theta_range, weights):
        sim_res = wf_model(all_x, all_y, type=type, ws=ws, wd=theta)
        flow_map = sim_res.flow_map(transect_grid, ws=ws, wd=theta)
        WS_eff = np.diag(flow_map.WS_eff.squeeze())
        weighted_ws_sum += weight * WS_eff

    return weighted_ws_sum / ws  # Normalize by inflow

def gaussian_filtered_ws(wf_model, all_x, all_y, type, ws, theta_range, weights):
    U_eff_sum = 0

    for theta, weight in zip(theta_range, weights):
        sim_res = wf_model(all_x, all_y, type=type, ws=ws, wd=theta)
        U_eff = sim_res.WS_eff.squeeze()
        U_eff_sum += weight * U_eff

    return U_eff_sum


def rotate_coords(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

def rotate_grid(all_x, all_y, wd, margin=1000, resolution=100):
    all_x_rot, all_y_rot = rotate_coords(all_x, all_y, 270 - wd)

    x_min, x_max = np.min(all_x_rot)-margin, np.max(all_x_rot)+margin
    y_min, y_max = np.min(all_y_rot)-margin, np.max(all_y_rot)+margin

    grid_x = np.linspace(x_min, x_max, resolution)
    grid_y = np.linspace(y_min, y_max, resolution)

    return grid_x, grid_y

Du_wt_x, Du_wt_y = np.load("../wind_turbine_layouts/dudgeon_layout.npy")
shs_wt_x, shs_wt_y = np.load("../wind_turbine_layouts/sheringham_shoal_layout.npy")

site = Dudgeon()
ws = 8
wd = 235



### Front Row WTS, and the Transect
t = [66, 65, 64, 63, 60, 59, 54, 53, 46, 45, 34, 33]
labels = ["J05", "J04", "K05", "L05", "L04", "L03", "L02", "A01", "A02", "A03", "A04", "A05"]

selected_point = [[], []]

for idx in t:
    selected_point[0].append(Du_wt_x[idx])
    selected_point[1].append(Du_wt_y[idx])

# Turbine A05 coordinates
x_A05, y_A05 = Du_wt_x[33], Du_wt_y[33]

# Calculate upstream point (1.4 km upstream of inflow at 235°)
distance_upstream = 1400  # 1.4 km
n_points = 15

x_transect_center = x_A05 - distance_upstream / math.sqrt(2)
y_transect_center = y_A05 - distance_upstream / math.sqrt(2)

# Define the perpendicular direction to inflow (325°)
angle_transect = np.deg2rad(235 + 70)

# Transect length is 14 km; since we have 15 points, spacing is 1 km each
spacing = 940  # spacing = 1000 m (1 km)

# The center point is at the position of the 3rd topmost point (2 points above, 12 below)
points_above = 2
points_below = 12

# Calculate start and end points accordingly
x_start = x_transect_center - points_above * spacing * np.cos(angle_transect)
y_start = y_transect_center - points_above * spacing * np.sin(angle_transect)

x_end = x_transect_center + points_below * spacing * np.cos(angle_transect)
y_end = y_transect_center + points_below * spacing * np.sin(angle_transect)

# Create 15 transect points clearly spaced
transect_x = np.linspace(x_start, x_end, n_points)
transect_y = np.linspace(y_start, y_end, n_points)


wt = SWT_60_154()

windTurbines = WindTurbines.from_WindTurbine_lst([SWT_60_154(), SWT_36_107()])
windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]

deficitModels =  {
         'NOJ': [noj_WF_model(site, windTurbines), '#c65102', '--'],
          'Bastankhah': [bastankhah_WF_model(site, windTurbines), '#ff028d', '--'],
          'TurboNoj': [turboNoj_WF_model(site, windTurbines), '#a8ff04', '--'],
          'SuperGaussian': [blondelSuperGaussian_WF_model(site, windTurbines), '#0c06f7', '--'],
          'Niayifar': [niayifar_WF_model(site, windTurbines), '#8cffdb', '--'],
          'Zong': [zong_WF_model(site, windTurbines), '#fffd01','--'],
          'Nygaard': [nygaard_WF_model(site, windTurbines), '#ff000d', '--'],
          'Carbajo': [carbajo_WF_model(site, windTurbines), '#653700', '--'],
        # 'Nygaard-Original': [nygaard_original_model(site, windTurbines), 'r', '-'],
        # 'Nygaard-Revised' :[nygaard_paul_revised_model(site, windTurbines), 'r', '--']
        }

all_x, all_y = np.r_[Du_wt_x, shs_wt_x], np.r_[Du_wt_y, shs_wt_y]
type = [0] * len(Du_wt_x) + [1] * len(shs_wt_x)
all_x = (all_x - selected_point[0][0])
all_y = (all_y - selected_point[1][0])

wf_model = nygaard_WF_model(site, windTurbines)
sim_res = wf_model(all_x, all_y, ws=ws, wd=wd)
# # Extract flow map at rotated grid positions
flow_map_default = sim_res.flow_map(ws=ws, wd=wd)

selected_wts = [(selected_point[0]-selected_point[0][0]), (selected_point[1]-selected_point[1][0])]
transect_x -= selected_point[0][0] 
transect_y -= selected_point[1][0]

###########################################################################
#### Below is the wake contour to get the number/index of the turbines ####
###########################################################################
# wind_turbine = SWT_60_154()

# wf_model = Nygaard_2022(site, wind_turbine)
# sim_res = wf_model(Du_wt_x, Du_wt_y)

# flow_map = sim_res.flow_map(grid=None, wd=235, ws=10)

# plt.figure()
# flow_map.plot_wake_map()
# plt.show()
# exit(0)
###########################################################################


###########################################################################
################       Wake Magnitude from WT power       #################
###########################################################################

# plt.figure(figsize=(9, 4))
# x_positions = np.arange(len(labels))

# for name, deficitModel in deficitModels.items():
#     #sim_res = deficitModels[deficitModel](all_x, all_y, type=type, ws=ws, wd=wd)
#     filtered_ws = gaussian_filtered_ws(deficitModel[0], all_x, all_y, type, ws, theta_range, weights)

#     #U_eff = sim_res.WS_eff.squeeze()

#     #front_row_ws = U_eff[t] # equivelant ws at front rows turbines
#     front_row_filt_ws = filtered_ws[t]

#     # Positions for evenly spacing the labels
#     # plt.plot(x_positions, front_row_ws/ws, '-', label=deficitModel, linewidth=2)
#     plt.plot(x_positions, front_row_filt_ws/ws, deficitModel[2], color = deficitModel[1], label=name + ', GA', linewidth=2.5)
# plt.plot(x_positions, RANS_AD_GA['wake_magnitude'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# plt.plot(x_positions, RANS_AD_AWF['wake_magnitude'], '-^', color='g', label='RANS-AWF, GA', linewidth=2.5, markersize=6 )
# plt.title('Wake Magnitude from WT Power', fontsize=18)
# plt.xlabel('Wind Turbine', fontsize=16)
# plt.ylabel(r'$U_{wt}/U_{\infty}$', fontsize=16)
# plt.xticks(x_positions, labels, fontsize=12)
# plt.yticks(np.arange(0.86, 1, 0.02), fontsize=12)
# #plt.ylim(0.86, 1)
# plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True, linestyle='--')
# plt.show()
###########################################################################


###########################################################################
################       Wake Shape from WT power       #################
###########################################################################
# plt.figure(figsize=(9, 4))
# for name, deficitModel in deficitModels.items():
#     #sim_res = deficitModels[deficitModel](all_x, all_y, type=type, ws=ws, wd=wd)
#     filtered_ws = gaussian_filtered_ws(deficitModel[0], all_x, all_y, type, ws, theta_range, weights)
#     #U_eff = sim_res.WS_eff.squeeze()
#     front_row_filt_ws = filtered_ws[t]

#     #front_row_ws = U_eff[t] # equivelant ws at front rows turbines
    
#     mean_front_filt_ws = np.mean(front_row_filt_ws)
#     #mean_front_ws = np.mean(front_row_ws) # # Normalize by mean effective wind speed of front row

#     #plt.plot(x_positions, front_row_ws/mean_front_ws, ('-')[i // 10], label=deficitModel, linewidth=2)
#     plt.plot(x_positions, front_row_filt_ws/mean_front_filt_ws, deficitModel[2], color=deficitModel[1] , label=name + ", GA", linewidth=2.5)

# plt.plot(x_positions, RANS_AD_GA['wake_shape'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# plt.plot(x_positions, RANS_AD_AWF['wake_shape'], '-^', color='g', label='RANS-AWF, GA', linewidth=2.5, markersize=6)
# plt.fill_between(x_positions, SCADA_DATA['lower'], SCADA_DATA['upper'], 
#                  color='gray',
#                  alpha=0.4, label='SCADA')
# plt.title('Wake Shape from WT Power', fontsize=18)
# plt.xlabel('Wind Turbine', fontsize=16)
# plt.ylabel(r'$U_{wt}/U_{ref}$', fontsize=16)
# plt.xticks(x_positions, labels, fontsize=12)
# plt.yticks(np.arange(0.94, 1.06, 0.02), fontsize=12)
# plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True, linestyle='--')
# plt.show()
###########################################################################

###########################################################################
############### Wake contour of Dudgeon vs Shenanigam Shoal ###############
###########################################################################
# # def wake_no_rot(sim_res):
# #     sim_res.flow_map().plot_wake_map(500, cmap='jet', plot_windturbines=False)

# #     plt.scatter(transect_x, 
# #                 transect_y,
# #                 marker='.',
# #                 facecolors='None',
# #                 edgecolors='w'
# #                 ,s=20)
    
# #     ax = plt.gca()
# #     plt.tick_params(axis='both', which='major', labelsize=12)
# #     plt.xlabel(r'$x - x_{A05} [km]$', fontsize=16)
# #     plt.ylabel(r'$y - y_{A05} [km]$', fontsize=16)
# #     plt.title("TurbOPark(Nygaard)")
# #     plt.tight_layout()
# #     plt.show()
# #     exit(0)

# # wake_no_rot(sim_res=sim_res)
######## HERE #########
# wf_model = nygaard_WF_model(site, windTurbines)
# sim_res = wf_model(all_x, all_y, ws=ws, wd=wd)
# # # Extract flow map at rotated grid positions

# # Default boundaries from PyWake
# x_min, x_max = flow_map_default.x.values.min(), flow_map_default.x.values.max()
# y_min, y_max = flow_map_default.y.values.min(), flow_map_default.y.values.max()

# # Slightly expand boundaries
# margin = 10000
# resolution = 750

# x_expanded = np.linspace(x_min - margin, x_max + margin, resolution)
# y_expanded = np.linspace(y_min - margin, y_max + margin, resolution)

# # Efficiently create expanded XYGrid
# expanded_grid = XYGrid(x=x_expanded, y=y_expanded)

# # Evaluate wake efficiently
# flow_map_expanded = sim_res.flow_map(expanded_grid, ws=ws, wd=wd)

# # Explicit centering of flow map coordinates
# X_exp, Y_exp = np.meshgrid(flow_map_expanded.x, flow_map_expanded.y)

# # Apply rotation for visualization
# X_rot, Y_rot = rotate_coords(X_exp, Y_exp, 360 - wd)
# selected_wts_rot_x, selected_wts_rot_y = rotate_coords(selected_wts[0], selected_wts[1], 360 - wd)
# transect_rot_x, transect_rot_y = rotate_coords(transect_x, transect_y, 360 - wd)

# X_rot_A05, Y_rot_A05 = selected_wts_rot_x[11], selected_wts_rot_y[11]

# selected_wts_rot_x -= X_rot_A05
# selected_wts_rot_y -= Y_rot_A05

# transect_rot_x -= X_rot_A05
# transect_rot_y -= Y_rot_A05

# X_rot -= X_rot_A05
# Y_rot -= Y_rot_A05

# # Define contour levels clearly
# levels = np.linspace(0.3, 1.05, 16)

# # Plot contour
# plt.figure()
# plt.contourf(X_rot, Y_rot, flow_map_expanded.WS_eff.squeeze() / ws,
#              levels=levels, cmap='jet', extend='min')

# # Colorbar clearly set
# ticks = np.linspace(0.3, 1.0, 8)
# cbar = plt.colorbar(label=r"$\sqrt{U^2 + V^2}/U_{\infty}$", ticks=ticks)
# cbar.ax.tick_params(labelsize=14)
# cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])

# # Plot turbine positions
# plt.scatter(selected_wts_rot_x, selected_wts_rot_y, marker='.',
#             facecolors='none', edgecolors='b', s=20)

# # Plot transect points
# plt.scatter(transect_rot_x, transect_rot_y, marker='o',
#             edgecolors='purple', s=18)

# # Plot formatting
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# ax.set_xticks([-10000, -5000, 0, 5000])
# ax.set_yticks([20000, 15000, 10000, 5000, 0, -5000, -10000])
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.xlabel(r'$x - x_{A05} [km]$', fontsize=16)
# plt.ylabel(r'$y - y_{A05} [km]$', fontsize=16)
# plt.title("TurbOPark - Nygaard", fontsize=16)
# plt.xlim(-14000, 7000)
# plt.ylim(-12000, 21000)
# scale_axis(ax, meter_to_km)
# ax.set_xticklabels(["-10", "-5", "0", "5"])
# ax.set_yticklabels(["-20","-15", "-10", "-5", "0", "5", "10"])
# plt.tight_layout()
# plt.show()
###########################################################################


###########################################################################
####### Scatter plot with annotation of the front row of Dudgeon ##########
###########################################################################
#D = SWT_60_154().diameter()

# Du_wt_x = (Du_wt_x - selected_point[0][0])/D
# Du_wt_y = (Du_wt_y - selected_point[1][0])/D

# selected_wts = [(selected_point[0]-selected_point[0][0])/D, (selected_point[1]-selected_point[1][0])/D]

# plt.figure(figsize=[6,8])
# plt.scatter(Du_wt_x, Du_wt_y, c ='k', marker='.',s=20)
# plt.scatter(selected_wts[0], 
#             selected_wts[1],
#             marker='o',
#             facecolors='None',
#             edgecolors='r'
#             ,s=17)

# for i in range(len(selected_wts[0])):
#     plt.annotate(labels[i], (selected_wts[0][i], selected_wts[1][i]),  # The point to annotate
#                  textcoords="offset points",  # How to position the text
#                  xytext=(0, 10),  # Distance from text to points (x,y)
#                  ha='center')  # Horizontal alignment
# plt.xlabel(r'$(x - x_{A05})/{D}$', fontsize=16)
# plt.ylabel(r'$(y - y_{A05})/{D}$', fontsize=16)
# plt.show()
###########################################################################



# ###########################################################################
# #######            Wake magnitude at transect region             ##########
# ###########################################################################

# # Clearly set transect grid points
# transect_grid = XYGrid(x=transect_x, y=transect_y)
# # Distance along transect clearly in km
# distance_along_transect = np.linspace(0, 14, n_points)

# plt.figure(figsize=(9, 4))

# for name, deficitModel in deficitModels.items():
#     wake_magnitude_filt = gaussian_filtered_ws_at_transect(
#         deficitModel[0], all_x, all_y, type, ws, transect_grid, theta_range, weights
#     )
#     sim_res = deficitModel[0](all_x, all_y, type=type, ws=ws, wd=wd, n_cpu=None)

#     # Clearly extract flow map at transect points
#     flow_map_transect = sim_res.flow_map(transect_grid, ws=ws, wd=wd)

#     WS_eff_transect = np.diag(flow_map_transect.WS_eff.squeeze())

#     # print(WS_eff_transect.shape)  # (15,) exactly as intended
#     # Get horizontal wind speed components clearly
#     wake_magnitude = WS_eff_transect / ws

#     # print(wake_magnitude)

#     #plt.plot(distance_along_transect, wake_magnitude, ('-')[i // 10], label=deficitModel, linewidth=4)
#     plt.plot(distance_along_transect, wake_magnitude_filt, deficitModel[2], color=deficitModel[1], label=name + ', GA', linewidth=2.5)

# plt.plot(distance_along_transect, RANS_AWF, '-s', color='m', label='RANS-AWF, GA', linewidth=2.5, markersize=6)
# plt.plot(distance_along_transect, RANS_AD_GA['wake_mag_up'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# plt.fill_between(distance_along_transect, WRF_DATA['lower'], WRF_DATA['upper'], 
#                  color='blue',
#                  alpha=0.4, label='WRF')
# plt.xlabel("Distance along transect [km]", fontsize=14)
# plt.ylabel(r"$\sqrt{U^2 + V^2}/U_{\infty}$", fontsize=16)
# plt.title("Wake Magnitude Upstream of Dudgeon", fontsize=18)
# plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(np.arange(0, 14.01, 1), fontsize=12)
# plt.xlim(0, 14)
# plt.yticks(np.arange(0.84, 1.02, 0.02), fontsize=12)
# plt.grid(True, linestyle='--')
# plt.tight_layout()
# plt.show()
# ###########################################################################


###########################################################################
#######            Wake magnitude at transect region  different TIs           ##########
###########################################################################
# 8 TI values for a 2x4 grid
# ti_values = np.arange(0.02, 0.14, 0.02)  # 0.02, 0.04, ..., 0.16

# transect_grid = XYGrid(x=transect_x, y=transect_y)
# distance_along_transect = np.linspace(0, 14, n_points)

# fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharey=True)
# axes = axes.ravel()

# legend_handles, legend_labels = None, None

# for idx, ti in enumerate(ti_values):
#     ax = axes[idx]
#     site = Dudgeon(ti=ti)
#     deficitModels =  {
#         'NOJ': [noj_WF_model(site, windTurbines), '#c65102', '--'],
#         'Bastankhah': [bastankhah_WF_model(site, windTurbines), '#ff028d', '--'],
#         'TurboNoj': [turboNoj_WF_model(site, windTurbines), '#a8ff04', '--'],
#         'SuperGaussian': [blondelSuperGaussian_WF_model(site, windTurbines), '#0c06f7', '--'],
#         'Niayifar': [niayifar_WF_model(site, windTurbines), '#8cffdb', '--'],
#         'Zong': [zong_WF_model(site, windTurbines), '#fffd01','--'],
#         'Nygaard': [nygaard_WF_model(site, windTurbines), '#ff000d', '--'],
#         'Carbajo': [carbajo_WF_model(site, windTurbines), '#653700', '--'],
#     }

#     for name, deficitModel in deficitModels.items():
#         wake_magnitude_filt = gaussian_filtered_ws_at_transect(
#             deficitModel[0], all_x, all_y, type, ws, transect_grid, theta_range, weights
#         )
#         sim_res = deficitModel[0](all_x, all_y, type=type, ws=ws, wd=wd, n_cpu=None)
#         flow_map_transect = sim_res.flow_map(transect_grid, ws=ws, wd=wd)
#         WS_eff_transect = np.diag(flow_map_transect.WS_eff.squeeze())
#         wake_magnitude = WS_eff_transect / ws  # kept in case needed elsewhere

#         ax.plot(
#             distance_along_transect,
#             wake_magnitude_filt,
#             deficitModel[2],
#             color=deficitModel[1],
#             label=name + ', GA',
#             linewidth=2.5
#         )

#     # Title as TI value
#     ax.set_title(f"TI = {ti:.0%}", fontsize=12)

#     # Set ticks/limits per-axis (shared, but explicit)
#     ax.set_xticks(np.arange(0, 14.01, 2))
#     ax.set_xlim(0, 14)
#     ax.set_yticks(np.arange(0.84, 1.02, 0.04))
#     ax.grid(True, linestyle='--')

#     # Capture legend handles once (first panel)
#     if legend_handles is None:
#         legend_handles, legend_labels = ax.get_legend_handles_labels()

# # Shared labels
# fig.supxlabel("Distance along transect [km]", fontsize=14)
# fig.supylabel(r"$\sqrt{U^2 + V^2}/U_{\infty}$", fontsize=14)

# # One shared legend on top
# if legend_handles:
#     fig.legend(
#         legend_handles,
#         legend_labels,
#         loc='upper center',
#         ncol=4,
#         fontsize=9,
#         frameon=False,
#     )

# # Adjust layout to leave space for top legend
# plt.tight_layout(rect=[0, 0, 1, 0.94])
# plt.show()

###########################################################################



###########################################################################
####################### COMBINED 2x2 SUBPLOT LAYOUT #######################
###########################################################################
# fig = plt.figure(figsize=(12, 8))
# gs = fig.add_gridspec(2, 2, height_ratios=[1,1], hspace=0.3, wspace=0.25)

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, :])

# # 1. Wake Magnitude from WT Power
# x_positions = np.arange(len(labels))
# for name, deficitModel in deficitModels.items():
#     filtered_ws = gaussian_filtered_ws(deficitModel[0], all_x, all_y, type, ws, theta_range, weights)
#     front_row_filt_ws = filtered_ws[t]
#     ax1.plot(x_positions, front_row_filt_ws/ws, deficitModel[2], color=deficitModel[1], label=name + ', GA', linewidth=2.5)
# ax1.plot(x_positions, RANS_AD_GA['wake_magnitude'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# ax1.plot(x_positions, RANS_AD_AWF['wake_magnitude'], '-^', color='g', label='RANS-AWF, GA', linewidth=2.5, markersize=6)
# ax1.set_title('Wake Magnitude from WT Power', fontsize=14, pad=10)
# ax1.set_xlabel('Wind Turbine', fontsize=12)
# ax1.set_ylabel(r'$U_{wt}/U_{\infty}$', fontsize=14)
# ax1.set_xticks(x_positions)
# ax1.set_xticklabels(labels, fontsize=12)
# ax1.set_yticks(np.arange(0.86, 1, 0.02))
# ax1.tick_params(axis='y', labelsize=12)
# ax1.grid(True, linestyle='--')

# # 2. Wake Shape from WT Power
# for name, deficitModel in deficitModels.items():
#     filtered_ws = gaussian_filtered_ws(deficitModel[0], all_x, all_y, type, ws, theta_range, weights)
#     front_row_filt_ws = filtered_ws[t]
#     mean_front_filt_ws = np.mean(front_row_filt_ws)
#     ax2.plot(x_positions, front_row_filt_ws/mean_front_filt_ws, deficitModel[2], color=deficitModel[1], label=name + ", GA", linewidth=2.5)
# ax2.plot(x_positions, RANS_AD_GA['wake_shape'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# ax2.plot(x_positions, RANS_AD_AWF['wake_shape'], '-^', color='g', label='RANS-AD-AWF, GA', linewidth=2.5, markersize=6)
# ax2.fill_between(x_positions, SCADA_DATA['lower'], SCADA_DATA['upper'], color='gray', alpha=0.4, label='SCADA')
# ax2.set_title('Wake Shape from WT Power', fontsize=14, pad=10)
# ax2.set_xlabel('Wind Turbine', fontsize=12)
# ax2.set_ylabel(r'$U_{wt}/U_{ref}$', fontsize=14)
# ax2.set_xticks(x_positions)
# ax2.set_xticklabels(labels, fontsize=12)
# ax2.set_yticks(np.arange(0.94, 1.06, 0.02))
# ax2.tick_params(axis='y', labelsize=12)
# ax2.grid(True, linestyle='--')

# # 3. Wake Magnitude Upstream of Dudgeon (transect)
# transect_grid = XYGrid(x=transect_x, y=transect_y)
# distance_along_transect = np.linspace(0, 14, n_points)

# for name, deficitModel in deficitModels.items():
#     wake_magnitude_filt = gaussian_filtered_ws_at_transect(
#         deficitModel[0], all_x, all_y, type, ws, transect_grid, theta_range, weights
#     )
#     ax3.plot(distance_along_transect, wake_magnitude_filt, deficitModel[2], color=deficitModel[1], label=name + ', GA', linewidth=2.5)
# ax3.plot(distance_along_transect, RANS_AWF, '-s', color='m', label='RANS-AWF, GA', linewidth=2.5, markersize=6)
# ax3.plot(distance_along_transect, RANS_AD_GA['wake_mag_up'], '-*', color='k', label='RANS-AD, GA', linewidth=2.5, markersize=6)
# ax3.fill_between(distance_along_transect, WRF_DATA['lower'], WRF_DATA['upper'], color='blue', alpha=0.4, label='WRF')
# ax3.set_xlabel("Distance along transect [km]", fontsize=12)
# ax3.set_ylabel(r"$\sqrt{U^2 + V^2}/U_{\infty}$", fontsize=14)
# ax3.set_title("Wake Magnitude Upstream of Dudgeon", fontsize=14, pad=10)
# ax3.set_xticks(np.arange(0, 14.01, 1))
# ax3.set_xlim(0, 14)
# ax3.set_yticks(np.arange(0.84, 1.02, 0.02))
# ax3.tick_params(axis='both', labelsize=12)
# ax3.grid(True, linestyle='--')

# # Combined Legend on Top
# handles, labels = [], []
# for ax in [ax1, ax2, ax3]:
#     h, l = ax.get_legend_handles_labels()
#     handles += h
#     labels += l
# by_label = dict(zip(labels, handles))  # Remove duplicates
# fig.legend(by_label.values(), by_label.keys(), loc='upper center', fontsize=11, ncol=6)
# plt.show()
###########################################################################

