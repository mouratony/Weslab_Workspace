import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from py_wake.wind_turbines import WindTurbines
from windFarms_windTurbines import *
from py_wake import XYGrid
import math


wt = SWT_60_154()

windTurbines = WindTurbines.from_WindTurbine_lst([SWT_60_154(), SWT_36_107()])
windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]

# Du_wt_x, Du_wt_y = np.load("wind_turbine_layouts/dudgeon_layout.npy")
Du_wt_x, Du_wt_y = np.load("/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/wind_turbine_layouts/dudgeon_layout.npy")

# /Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/wind_turbine_layouts/dudgeon_layout.npy
# shs_wt_x, shs_wt_y = np.load("wind_turbine_layouts/sheringham_shoal_layout.npy")

shs_wt_x, shs_wt_y = np.load("/Users/Antonio/Desktop/Research_Projects/Wind_Energy_lab_V1/Wind_Farms_Wake_Interactions/Examples/wind_turbine_layouts/sheringham_shoal_layout.npy")

site = Dudgeon()
site.plot_wd_distribution(ws_bins=[9, 12, 15])


all_x, all_y = np.r_[Du_wt_x, shs_wt_x], np.r_[Du_wt_y, shs_wt_y]
type = [0] * len(Du_wt_x) + [1] * len(shs_wt_x)


wfm_not_waked = {
            'NOJ': [noj_WF_model(site, wt), '#c65102', '--'],
          'Bastankhah': [bastankhah_WF_model(site, wt), '#ff028d', '--'],
          'TurboNoj': [turboNoj_WF_model(site, wt), '#a8ff04', '--'],
          'SuperGaussian': [blondelSuperGaussian_WF_model(site, wt), '#0c06f7', '--'],
          'Niayifar': [niayifar_WF_model(site, wt), '#8cffdb', '--'],
          'Zong': [zong_WF_model(site, wt), '#fffd01','--'],
          'Nygaard': [nygaard_WF_model(site, wt), '#ff000d', '--'],
          'Carbajo': [carbajo_WF_model(site, wt), '#653700', '--'],
        }


wfm_waked = {
            'NOJ': [noj_WF_model(site, windTurbines), '#c65102', '-'],
          'Bastankhah': [bastankhah_WF_model(site, windTurbines), '#ff028d', '-'],
          'TurboNoj': [turboNoj_WF_model(site, windTurbines), '#a8ff04', '-'],
          'SuperGaussian': [blondelSuperGaussian_WF_model(site, windTurbines), '#0c06f7', '-'],
          'Niayifar': [niayifar_WF_model(site, windTurbines), '#8cffdb', '-'],
          'Zong': [zong_WF_model(site, windTurbines), '#fffd01','-'],
          'Nygaard': [nygaard_WF_model(site, windTurbines), '#ff000d', '-'],
          'Carbajo': [carbajo_WF_model(site, windTurbines), '#653700', '-'],
        }
# sim_res = wfm(all_x, all_y)

# power_MW = []
wind_speeds = [9, 12, 15] # m/s
wd_prevailing = 235

wd_deg = np.arange(0, 360, 1)
# bins_wd = {
#         '225-270':slice(225, 270),
#         '0-45':slice(0, 45),
#         '45-90':slice(45, 90),
#         '90-135':slice(90, 135),
#         '135-180':slice(135, 180),
#         '180-225':slice(180, 225),
#         '225-270':slice(225, 270),
#         '270-315':slice(270, 315),
#         '315-360':slice(315, 360)
#     }

# 1) Grid: rows = ws cases, cols = fixed model groups; share x only
nrows, ncols = len(wind_speeds), 3
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(16, 9))
if axes.ndim == 1:
    axes = axes.reshape(nrows, ncols)

# Column -> model groups (fixed per column)
col_groups = [
    ['NOJ', 'Bastankhah', 'TurboNoj'],
    ['Zong', 'Niayifar', 'Carbajo'],
    ['Nygaard', 'SuperGaussian'],
]

# Helper: set consistent tick count (same number across all y-axes)
# def set_equal_tick_count(ax, ymin, ymax, n=5):
#     if ymin == ymax:
#         ymin, ymax = ymin * 0.95, ymax * 1.05 if ymax != 0 else (-0.5, 0.5)
#     ticks = np.linspace(ymin, ymax, num=n)
#     ax.set_yticks(ticks)

all_handles, all_labels = [], []

for r, ws in enumerate(wind_speeds):
    for c, models in enumerate(col_groups):
        ax = axes[r, c]

        yvals = []  # collect to compute local min/max for this subplot

        for k in models:
            sim_res_not_waked = wfm_not_waked[k][0](Du_wt_x, Du_wt_y, n_cpu=None)
            sim_res_waked     = wfm_waked[k][0](all_x, all_y, type=type, n_cpu=None)

            ccol      = wfm_not_waked[k][1]
            marker_nw = wfm_not_waked[k][2]
            marker_w  = wfm_waked[k][2]


            # Normalize the Power as a ratio of the Power at prevailing wind direction

            # p_not_waked_prev = sim_res_not_waked.Power.isel(wd=wd_prevailing, ws=ws - 3).sum().values / 1e6
            # p_waked_prev = sim_res_waked.Power.isel(wd=wd_prevailing, ws=ws - 3).sum().values / 1e6
            p_not_waked = [sim_res_not_waked.Power.isel(wd=wd, ws=ws - 3).sum().values / 1e6 for wd in wd_deg]  # MW
            p_waked     = [sim_res_waked.Power.isel(wd=wd, ws=ws - 3, wt=np.arange(len(Du_wt_x))).sum().values / 1e6 for wd in wd_deg]
            

            p_not_waked_norm = p_not_waked /max(p_not_waked)
            # print(max(p_not_waked))
            p_waked_norm = p_waked / max(p_waked)


            # ln1, = ax.plot(wd_deg, p_not_waked_norm, ls=marker_nw, c=ccol, lw=1.4, label=f"{k} (Not Waked)")
            # ln2, = ax.plot(wd_deg, p_waked_norm,     ls=marker_w,  c=ccol, lw=1.4, label=f"{k} (Waked)")
            ln1, = ax.plot(wd_deg, p_not_waked_norm, ls=marker_nw, c=ccol, lw=1.4, label=f"{k} (Not Waked)")
            ln2, = ax.plot(wd_deg, p_waked_norm,     ls=marker_w,  c=ccol, lw=1.4, label=f"{k} (Waked)")

            yvals.extend(p_not_waked)
            yvals.extend(p_waked)

            # collect legend entries from first row only (to avoid dupes)
            if r == 0:
                all_handles.extend([ln1, ln2])
                all_labels.extend([ln1.get_label(), ln2.get_label()])

        # 2) Good y-axis: local min/max with small margin, same tick count
        # y_min, y_max = np.min(yvals), np.max(yvals)
        # pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * (abs(y_max) + 1)
        # ax.set_ylim(y_min - pad, y_max + pad)
        # set_equal_tick_count(ax, ax.get_ylim()[0], ax.get_ylim()[1], n=5)
        # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))

        # 3) Row title: same ws across all 3 plots in the row
        if c == 0:
            ax.set_title(f"ws = {ws} m/s", fontsize=11, loc='left')

        ax.grid(True, alpha=0.25)

# 4) Global axis labels (single y label for all, shared x)
fig.supxlabel("Wind direction (deg)", fontsize=11)
fig.supylabel("Power (MW)", fontsize=11)

# 5) Top legend with 8 columns, dedup handles while preserving order
seen = set()
uniq_handles, uniq_labels = [], []
for h, lab in zip(all_handles, all_labels):
    if lab not in seen:
        uniq_handles.append(h)
        uniq_labels.append(lab)
        seen.add(lab)

fig.legend(
    handles=uniq_handles,
    labels=uniq_labels,
    loc='upper center',
    ncol=8,
    bbox_to_anchor=(0.5, 1.0),
    frameon=False,
    fontsize=6,
    handlelength=2
)

fig.suptitle("Power Loss of Dudgeon across different wd and ws", y=1.05, fontsize=14)
fig.subplots_adjust(top=0.82, wspace=0.25, hspace=0.35)
plt.show()

# power_MW = []
# wd_deg = np.arange(0, 360, 1)
# for wd in wd_deg:
#     sim_res = wfm(Du_wt_x, Du_wt_y, wd = wd)
#     # sim_res = wfm(all_x, all_y, wd = wd)
#     power_MW.append(sim_res.Power.sum().values/1e6) # MW

# plt.figure()
# plt.plot(wd_deg, power_MW)
# plt.xlabel("wd (deg)")
# plt.ylabel("Power (MW)")
# plt.title("Power Produce by the Farm per wd")
# plt.tight_layout()
# sim_res = wfm(Du_wt_x, Du_wt_y)
# print(sim_res)
# num = (sim_res.Power * sim_res.P).sum(dim=['ws', 'wt']) 
# # denominator: P(wd) = sum_ws P(wd,ws)
# den = sim_res.P.sum(dim='ws')

# E_power_wd_MW = (num / den) /1e6     

# wd_deg = E_power_wd_MW['wd'].values
# y = E_power_wd_MW.values


# plt.figure()
# plt.plot(wd_deg, y)
# plt.xlabel("wd (deg)")
# plt.ylabel("Expected Power")
# plt.title("E[Power | wd]")
# plt.tight_layout()
# plt.show()



