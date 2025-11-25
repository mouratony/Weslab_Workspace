import sys
sys.path.append('/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/') 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from py_wake.wind_turbines import WindTurbines
from py_wake.deficit_models.no_wake import NoWakeDeficit
from windFarms_windTurbines import *
from py_wake import XYGrid, Points
from scipy.stats import norm
from scipy.ndimage import convolve1d

site = Rodsand_2()
ws_cases = [9.0, 10.0, 11.0, 12.0] # clearly match paper
wd_cases = np.arange(0, 360, 1)  
windTurbines = WindTurbines.from_WindTurbine_lst([SWT_23_93(), SWT_23_82()])
windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]

# RANS_ABL_f1 = {'122D': (
#         np.array([9.3, 9.0, 8.7, 8.4, 8.1, 7.8, 7.5, 7.2]),   # x (WS_hub)
#         np.array([60, 40, 20,   0, -20, -40, -60, -80])      # y (S-N [D])
#     ),
# }

# RANS_ABL_f2 = {
#     '122D': (
#         np.array([9.6, 9.3, 9.1, 8.9, 8.6, 8.3, 8.0, 7.7]),   # x (WS_hub)
#         np.array([60, 40, 20,   0, -20, -40, -60, -80])      # y (S-N [D])
#     )
# }

SCADA_f1_13D = {
    'y': np.array([ -9,  -1,  4, 11]),
    'x_mean': np.array([-2.9, -3.1, -3.2, -3.3]),
    'x_err_lower': np.array([2.8, 2.5, 2.35, 2.6]),
    'x_err_upper': np.array([2.6, 2.5, 2.3, 2.5])
}


SCADA_f2_13D = {
    'y': np.array([ -9,  -1,  4, 11]),
    'x_mean': np.array([-2.75, -2.7, -2.25, -2.1]),
    'x_err_lower': np.array([1.3, 1.2, 1.25, 1.4]),
    'x_err_upper': np.array([1.3, 1.2, 1.2, 1.3])
}

SCADA_f1_35D = {
    'y':           np.array([-25,  -15,  -5,   -1,  7]),
    'x_mean':     np.array([ -3,  -2.9,  -3.3,  -2.95,  -2.75]),
    'x_err_lower':np.array([ 2.75,  2.4,  2.05,  2.55,  2.55]),
    'x_err_upper':np.array([ 2.6,  2.3,  2,  2.45,  2.55])
}

SCADA_f2_35D = {
    'y':           np.array([-25,  -15,  -5,   -1,  7]),
    'x_mean':     np.array([ -2.35,  -2.4,  -2.5,  -2.1,  -1.4]),
    'x_err_lower':np.array([ 1.65,  1.5,  1.05,  1.35,  1.6]),
    'x_err_upper':np.array([ 1.5,  1.45,  1.0,  1.2,  1.6])
}

deficitModels =  {
         'NOJ': [noj_WF_model(site, windTurbines), '#c65102', '--'],
          'Bastankhah': [bastankhah_WF_model(site, windTurbines), '#ff028d', '--'],
          'TurboNoj': [turboNoj_WF_model(site, windTurbines), '#FFBAF1', '--'],
          'SuperGaussian': [blondelSuperGaussian_WF_model(site, windTurbines), '#0c06f7', '-'],
          'Niayifar': [niayifar_WF_model(site, windTurbines), '#A00cdb', '--'],
          'Zong': [zong_WF_model(site, windTurbines), '#fffd01','--'],
          'Nygaard': [nygaard_WF_model(site, windTurbines), '#ff000d', '-'],
          'Carbajo': [carbajo_WF_model(site, windTurbines), '#653700', '--'],
        }
wt = SWT_23_93()
D = wt.diameter()

Ny_wt_x, Ny_wt_y = np.load("../wind_turbine_layouts/nysted_layout.npy")
Ro_wt_x, Ro_wt_y = np.load("../wind_turbine_layouts/rodsand2_layout.npy")
selected_point = [Ro_wt_x[4], Ro_wt_y[4]]

Ro_wt_x_m = (Ro_wt_x - selected_point[0])
Ro_wt_y_m = (Ro_wt_y - selected_point[1])
Ny_wt_x_m = (Ny_wt_x - selected_point[0])
Ny_wt_y_m = (Ny_wt_y - selected_point[1])
Ro_wt_x = Ro_wt_x_m/D
Ro_wt_y = Ro_wt_y_m/D
Ny_wt_x = Ny_wt_x_m/D
Ny_wt_y = Ny_wt_y_m/D


transect_y_D = np.linspace(-80, 80, 100)

# --- Main plot ---
all_x, all_y = np.r_[Ro_wt_x_m, Ny_wt_x_m], np.r_[Ro_wt_y_m, Ny_wt_y_m]
types = [0]*len(Ro_wt_x) + [1]*len(Ny_wt_x)

transect_x_pos = [230, 144, 122, 35, 13, -9]

# Gaussian weights (5° std, centered at 0° for circular convolution)
wd_gaussian_weights = norm.pdf(np.arange(-180,180), loc=0, scale=5)
wd_gaussian_weights /= wd_gaussian_weights.sum()

fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

legend_handles, legend_labels = [], []

for i, t_x_p in enumerate(transect_x_pos):
    ax = axes.flatten()[i]

    transect_y_m = np.linspace(-80*D, 80*D, 100)
    transect_x_m = t_x_p * D
    transect_grid = Points(x=[transect_x_m]*len(transect_y_m), 
                           y=transect_y_m, 
                           h=[69]*len(transect_y_m))

    for name, deficitModel in deficitModels.items():
        res_nyro = deficitModel[0](x=all_x, y=all_y, type=types, wd=wd_cases, ws=ws_cases)
        flow_nyro = res_nyro.flow_map(transect_grid)

        WS_NYRØ = flow_nyro.WS_eff.squeeze().values
        WS_NYRØ_gaussian = np.zeros_like(WS_NYRØ)

        for ws_idx in range(len(ws_cases)):
            for pt_idx in range(len(transect_y_m)):
                WS_NYRØ_gaussian[pt_idx, :, ws_idx] = convolve1d(
                    WS_NYRØ[pt_idx, :, ws_idx], wd_gaussian_weights, mode='wrap')

        wd_sector = (wd_cases >= 82) & (wd_cases <= 98)
        WS_NYRØ_linear = WS_NYRØ_gaussian[:, wd_sector, :].mean(axis=1)
        WS_NYRØ_final = WS_NYRØ_linear.mean(axis=1)
        WS_deficit_final = WS_NYRØ_final - 10

        line, = ax.plot(WS_deficit_final, transect_y_m/D, deficitModel[2], 
                        linewidth=1.5, color=deficitModel[1], label=name)

        # Save handles for legend explicitly from transect 13D
        if t_x_p == 13:
            legend_handles.append(line)
            legend_labels.append(name)

    # SCADA data explicitly for transects 13D and 35D
    if t_x_p == 13:
        sc1 = ax.errorbar(SCADA_f1_13D['x_mean'], SCADA_f1_13D['y'],
                    xerr=np.vstack([SCADA_f1_13D['x_err_lower'], SCADA_f1_13D['x_err_upper']]),
                    fmt='o', elinewidth=2, ecolor='c', color='c', capsize=4, linewidth=2, alpha=0.6, label='SCADA f1')
        sc2 = ax.errorbar(SCADA_f2_13D['x_mean'], SCADA_f2_13D['y'],
                    xerr=np.vstack([SCADA_f2_13D['x_err_lower'], SCADA_f2_13D['x_err_upper']]),
                    fmt='o', elinewidth=2, ecolor='g', color='g', capsize=4, linewidth=2, alpha=0.6, label='SCADA f2')
        
        # Explicitly add SCADA handles to the legend
        legend_handles.extend([sc1, sc2])
        legend_labels.extend(['SCADA f1', 'SCADA f2'])

    if t_x_p == 35:
        ax.errorbar(SCADA_f1_35D['x_mean'], SCADA_f1_35D['y'],
                    xerr=np.vstack([SCADA_f1_35D['x_err_lower'], SCADA_f1_35D['x_err_upper']]),
                    fmt='o', elinewidth=2 , ecolor='c', color='c', capsize=4, linewidth=2, alpha=0.6)
        ax.errorbar(SCADA_f2_35D['x_mean'], SCADA_f2_35D['y'],
                    xerr=np.vstack([SCADA_f2_35D['x_err_lower'], SCADA_f2_35D['x_err_upper']]),
                    fmt='o', elinewidth=2, ecolor='g', color='g', capsize=4, linewidth=2, alpha=0.6)

    ax.set_xlabel(r"$WS_{wf,hub} - WS_{NWF,hub}\,[m/s]$", fontsize=9)
    ax.set_ylabel("Turbine Spacing S-N [D]", fontsize=9)
    ax.set_title(f"Transect: {t_x_p}D", fontsize=10)
    ax.set_xlim(-6, 0.5)
    ax.set_ylim(-60, 55)
    ax.grid(True)

    # Adjusted inset plot position and smaller scatter markers explicitly
    ax_inset = ax.inset_axes([0.7, 0.75, 0.25, 0.2])
    ax_inset.grid(True)
    ax_inset.scatter(Ro_wt_x, Ro_wt_y, marker='.', s=8)
    ax_inset.scatter(Ny_wt_x, Ny_wt_y, color='tab:orange', marker='.', s=8)
    ax_inset.plot([t_x_p]*len(transect_y_D), transect_y_D, color='tab:olive', linewidth=1)
    ax_inset.tick_params(axis='both', labelsize=6)
    ax_inset.set_aspect('equal', adjustable='box')
    ax_inset.set_xlim(-20, 250)
    ax_inset.set_ylim(-60, 52)

    print(f"Transect {t_x_p}D Processed Successfully")

# Single legend explicitly from transect 13D including SCADA
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), 
           fontsize=8, ncol=5, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.3, wspace=0.25)

plt.show()




# transect_grid = Points(x=[transect_x_m]*len(transect_y_m), 
#                        y=transect_y_m, 
#                        h=[69]*len(transect_y_m))

# # Gaussian weights (5° std, centered at 0° for circular convolution)
# wd_gaussian_weights = norm.pdf(np.arange(-180,180), loc=0, scale=5)
# wd_gaussian_weights /= wd_gaussian_weights.sum()

# fig, ax_main = plt.subplots(figsize=(10, 6))

# # Clearly loop through all wake deficit models
# for name, deficitModel in deficitModels.items():

#     # --- Run full 360° simulation (NYRØ scenario) ---
#     res_nyro = deficitModel(x=all_x, y=all_y, type=types, wd=wd_cases, ws=ws_cases)
#     flow_nyro = res_nyro.flow_map(transect_grid)

#     # Ensure your data is clearly in numpy array explicitly:
#     # WS_RØ = flow_ro.WS_eff.squeeze().values  # shape explicitly (100, 360, 4)
#     WS_NYRØ = flow_nyro.WS_eff.squeeze().values

#     # Explicitly define arrays for filtered results clearly stated:
#     # WS_RØ_gaussian = np.zeros_like(WS_RØ)      # (100,360,4)
#     WS_NYRØ_gaussian = np.zeros_like(WS_NYRØ)  # (100,360,4)

#     # Gaussian convolution explicitly and clearly along WD axis:
#     for ws_idx in range(len(ws_cases)):
#         for pt_idx in range(len(transect_y_m)):
#             # WS_RØ_gaussian[pt_idx, :, ws_idx] = convolve1d(
#             #     WS_RØ[pt_idx, :, ws_idx], wd_gaussian_weights, mode='wrap')

#             WS_NYRØ_gaussian[pt_idx, :, ws_idx] = convolve1d(
#                 WS_NYRØ[pt_idx, :, ws_idx], wd_gaussian_weights, mode='wrap')
            
#     # print("WS_RØ_gaussian shape explicitly:", WS_RØ_gaussian.shape)
#     print(f'Model {name} has been processed!')
#     # --- Linear averaging explicitly within 82°–98° sector ---
#     wd_sector = (wd_cases >= 82) & (wd_cases <= 98)  # correct: length = 360 exactly

#     # clearly correct indexing explicitly stated:
#     # WS_RØ_linear = WS_RØ_gaussian[:, wd_sector, :].mean(axis=1)  # shape clearly (100,4)
#     WS_NYRØ_linear = WS_NYRØ_gaussian[:, wd_sector, :].mean(axis=1)

#     # WS_RØ_final = WS_RØ_linear.mean(axis=1)  # clearly (100,)
#     WS_NYRØ_final = WS_NYRØ_linear.mean(axis=1)

#     WS_deficit_final = WS_NYRØ_final - 10


#     # Plotting explicitly clearly matching paper
#     ax_main.plot(WS_deficit_final, transect_y_m/D, '-', label=name, linewidth=1.5)

# # SCADA clearly plotted alongside models
# ax_main.errorbar(SCADA_f1_35D['x_mean'], SCADA_f1_35D['y'],
#                  xerr=np.vstack([SCADA_f1_35D['x_err_lower'], SCADA_f1_35D['x_err_upper']]),
#                  fmt='o', elinewidth=2.5 , ecolor='c', color='c', capsize=5, linewidth=3, label='SCADA f1', alpha=0.6)

# ax_main.errorbar(SCADA_f2_35D['x_mean'], SCADA_f2_35D['y'],
#                  xerr=np.vstack([SCADA_f2_35D['x_err_lower'], SCADA_f2_35D['x_err_upper']]),
#                  fmt='o', elinewidth=2.5, ecolor='g', color='g', capsize=5, linewidth=3, label='SCADA f2', alpha=0.6)

# # Customizing ticks to match paper
# ax_main.set_xticks(np.arange(-6, 1, 2))
# ax_main.set_yticks(np.arange(-50, 51, 50))

# # Enhancing readability
# ax_main.tick_params(axis='both', direction='in', length=6, width=1, labelsize=12)
# ax_main.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)

# ax_main.set_xlabel(r"$WS_{wf,hub} - WS_{NWF,hub}\,[m/s]$", fontsize=14)
# ax_main.set_ylabel("Turbine Spacing S-N [D]", fontsize=14)
# ax_main.set_title(f"Transect: {t_x_p}D", fontsize=16)
# ax_main.set_xlim(-6, 0.5)
# ax_main.set_ylim(-60, 55)
# ax_main.grid(True)
# ax_main.legend(fontsize=13, bbox_to_anchor=(1.05, 1), loc='upper left')

# # Inset clearly maintained as in your original code (unchanged)
# ax_inset = ax_main.inset_axes([0.05, 0.75, 0.25, 0.25])
# ax_inset.grid(True)
# ax_inset.scatter(Ro_wt_x, Ro_wt_y, marker='.', s=15, label='Rodsand')
# ax_inset.scatter(Ny_wt_x, Ny_wt_y, color='tab:orange', marker='.', s=15, label='Nysted')
# ax_inset.plot(transect_x_D, transect_y_D, color='tab:olive', linewidth=2)
# ax_inset.tick_params(axis='both', which='major', labelsize=8)
# ax_inset.set_aspect('equal', adjustable='box')
# ax_inset.set_xlim(-20, 250)
# ax_inset.set_ylim(-60, 52)

# plt.tight_layout()
# plt.show()

# # --- Region for histogram (Rødsand II area) ---
# region_grid = XYGrid(
#     x=np.linspace(min(Ro_wt_x_m)-1000, max(Ro_wt_x_m)+1000, 100),
#     y=np.linspace(min(Ro_wt_y_m)-1000, max(Ro_wt_y_m)+1000, 100),
#     h=69
# )

# # --- Baseline scenario (Rødsand II only) ---
# wf_model_no_wake = PropagateDownwind(site, wt, wake_deficitModel=NoWakeDeficit())
# sim_base = wf_model_no_wake(Ro_wt_x_m, Ro_wt_y_m, type=0, ws=ws, wd=wd)
# flow_base = sim_base.flow_map(region_grid)
# WS_base = flow_base.WS_eff.squeeze()
# P_base = wt.power(WS_base)

# # --- Initialize arrays to store results ---
# WS_diff_abs = {}
# WS_diff_rel = {}
# P_diff_abs = {}
# P_diff_rel = {}

# for deficitModel, wfm in deficitModels.items():
#     sim_NYRO = wfm(all_x, all_y, type=types, ws=ws, wd=wd)
#     flow_NYRO = sim_NYRO.flow_map(region_grid)

#     WS_NYRO = flow_NYRO.WS_eff.squeeze()
#     P_NYRO = wt.power(WS_NYRO)

#     # --- Absolute WS and Power differences ---
#     WS_diff_abs[deficitModel] = (WS_NYRO - WS_base).values.flatten()
#     P_diff_abs[deficitModel] = (P_NYRO - P_base).flatten()

#     # --- Relative WS and Power differences ---
#     WS_diff_rel[deficitModel] = ((WS_NYRO - WS_base) / WS_base * 100).values.flatten()
#     P_diff_rel[deficitModel] = ((P_NYRO - P_base) / P_base * 100).flatten()



# # --- Plotting Histograms ---
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # --- Absolute Wind Speed ---
# ax_ws_abs = axes[0, 0]
# for model, values in WS_diff_abs.items():
#     ax_ws_abs.hist(values, bins=30, alpha=0.5, density=True, label=model)
# ax_ws_abs.set_xlabel(r'$WS_{hub,NYRØ}-WS_{hub,RØ}$ [m/s]', fontsize=13)
# ax_ws_abs.set_ylabel('Density')
# ax_ws_abs.grid(True)

# # --- Relative Wind Speed ---
# ax_ws_rel = axes[0, 1]
# for model, values in WS_diff_rel.items():
#     ax_ws_rel.hist(values, bins=30, alpha=0.5, density=True, label=model)
# ax_ws_rel.set_xlabel(r'$(WS_{hub,NYRØ}-WS_{hub,RØ})/WS_{hub,RØ}$ [%]', fontsize=13)
# ax_ws_rel.set_ylabel('Density')
# ax_ws_rel.grid(True)

# # --- Absolute Power ---
# ax_p_abs = axes[1, 0]
# for model, values in P_diff_abs.items():
#     ax_p_abs.hist(values, bins=30, alpha=0.5, density=True, label=model)
# ax_p_abs.set_xlabel(r'$P_{hub,NYRØ}-P_{hub,RØ}$ [kW]', fontsize=13)
# ax_p_abs.set_ylabel('Density')
# ax_p_abs.grid(True)

# # --- Relative Power ---
# ax_p_rel = axes[1, 1]
# for model, values in P_diff_rel.items():
#     ax_p_rel.hist(values, bins=30, alpha=0.5, density=True, label=model)
# ax_p_rel.set_xlabel(r'$(P_{hub,NYRØ}-P_{hub,RØ})/P_{hub,RØ}$ [%]', fontsize=13)
# ax_p_rel.set_ylabel('Density')
# ax_p_rel.grid(True)

# # --- Legend ---
# handles, labels = ax_p_rel.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12)

# plt.tight_layout(rect=[0, 0.05, 1, 0.98])
# plt.show()

