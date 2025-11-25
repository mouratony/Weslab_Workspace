import py_wake
import numpy as np
import matplotlib.pyplot as plt
from py_wake.wind_turbines import WindTurbines
import sys
import os
from tqdm import tqdm
import random
import math
from shapely.geometry import Point, Polygon
from shapely.affinity import translate
# Adjust the path to where your Examples directory is located
from windFarms_windTurbines import *
from cluster_analysis import *
from layout_dev import *
from aep_percent_diff_Plot import *

_ID_ = random.randint(0, 1000)

title_fontsize = 18
label_fontsize = 14
legend_fontsize = 12
axis_tick_size = 10

# # --- inputs you can adjust ---
# filepath = "DigitizeLayers/Europe/sheringham_shoal.geojson"
# area_km2 = 34.59
# rated_power_mw = 3.6
# capacity_densities = [3, 4, 5, 6]  # MW/km² (edit this list as needed)
# spacing_D = 6  # rotor-diameter spacing for your placement routine
# # -----------------------------

# # Boundary
# eastings, northings = get_only_boundary(filepath)
# eastings = np.array(eastings)
# northings = np.array(northings)
# boundary_points = list(zip(eastings, northings))

# # Figure grid
# n = len(capacity_densities)
# ncols = min(2, n)  # up to 3 per row; change if you prefer
# nrows = math.ceil(n / ncols)
# fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
# axes = np.atleast_1d(axes).ravel()

# # Common limits so all panels share view
# xmin, xmax = eastings.min(), eastings.max()
# ymin, ymax = northings.min(), northings.max()
# dx, dy = xmax - xmin, ymax - ymin
# padx, pady = 0.05*dx, 0.05*dy

# for ax, cd in zip(axes, capacity_densities):
#     # Compute number of turbines for this capacity density
#     nturbs = compute_number_of_turbines(area=area_km2, rated_power=rated_power_mw, capacity_density=cd)

#     # Generate turbine positions
#     tx, ty = grid_WTposition_generator(boundary_points, SWT_36_107(), nturbs, spacing=spacing_D)

#     # Plot boundary and turbines
#     ax.plot(eastings, northings, '-k', lw=1.2)
#     ax.plot(tx, ty, linestyle='none', marker='2', color='k', markersize=6)

#     # Formatting
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim(xmin - padx, xmax + padx)
#     ax.set_ylim(ymin - pady, ymax + pady)
#     ax.set_xlabel("X-UTM Coordinates [m]", fontsize=label_fontsize)
#     ax.set_ylabel("Y-UTM Coordinates [m]", fontsize=label_fontsize)
#     ax.tick_params(axis='both', labelsize=axis_tick_size)
#     ax.set_title(f"{nturbs} turbines | {cd} MW/km²", fontsize=12)

# # Hide any unused axes
# for ax in axes[len(capacity_densities):]:
#     ax.axis('off')

# fig.suptitle("Sheringham Shoal - Layout for Different Capacity Density", fontsize=title_fontsize)
# plt.tight_layout()
# plt.show()
# exit(0)

# def deficit_models_AEP(site, windTurbines, all_x, all_y, wt_x, wt_y, types = None, AEP_calc = True):

#     deficit_models_AEP = {}
#     noj_model = noj_WF_model(site, windTurbines)
#     turboNoj_model = turboNoj_WF_model(site, windTurbines)
#     fuga_model = fuga_WF_model(site, windTurbines)                                 
#     bastankhah_model = bastankhah_WF_model(site, windTurbines)
#     zong_model = zong_WF_model(site, windTurbines)
#     niayifar_model = niayifar_WF_model(site, windTurbines)
#     turboGaussian_model = turboGaussian_WF_model(site, windTurbines)
#     superGaussian_model = blondelSuperGaussian_WF_model(site, windTurbines)



#     models = []
#     models.append(noj_model)
#     models.append(turboNoj_model)
#     models.append(fuga_model)
#     models.append(bastankhah_model)
#     models.append(zong_model)
#     models.append(niayifar_model)
#     models.append(turboGaussian_model)
#     models.append(superGaussian_model)

#     if AEP_calc :

#         deficit_models_AEP['NOJ Model'] = (compute_AEP(noj_model, all_x, all_y, wt_x, wt_y, type=types),
#                                        percent_change_(noj_model, all_x, all_y, wt_x, wt_y, type=types)
#                                        )
        
#         deficit_models_AEP['TurboNOJ Model'] = (compute_AEP(turboNoj_model, all_x, all_y, wt_x, wt_y, type=types), 
#                                         percent_change_(turboNoj_model, all_x, all_y, wt_x, wt_y, type=types)
#                                         )
                                            
#         deficit_models_AEP['Fuga Model'] = (compute_AEP(fuga_model, all_x, all_y, wt_x, wt_y, type=types),
#                                             percent_change_(fuga_model, all_x, all_y, wt_x, wt_y, type=types)
#                                             )
        
#         deficit_models_AEP['Bastankhah Model'] = (compute_AEP(bastankhah_model, all_x, all_y, wt_x, wt_y, type=types), 
#                                                 percent_change_(bastankhah_model, all_x, all_y, wt_x, wt_y, type=types)  
#                                                 )
        
#         deficit_models_AEP['Zong Model'] = (compute_AEP(zong_model, all_x, all_y, wt_x, wt_y, type=types),
#                                             percent_change_(zong_model, all_x, all_y, wt_x, wt_y, type=types)
#                                             )

#         deficit_models_AEP['Niayifar Model'] = (compute_AEP(niayifar_model, all_x, all_y, wt_x, wt_y, type=types),
#                                                 percent_change_(niayifar_model, all_x, all_y, wt_x, wt_y, type=types)
#                                                 )
        
#         deficit_models_AEP['TurboGaussian Model'] = (compute_AEP(turboGaussian_model, all_x, all_y, wt_x, wt_y, type=types),
#                                                     percent_change_(turboGaussian_model, all_x, all_y, wt_x, wt_y, type=types)
#                                                     )
        
#         deficit_models_AEP['SuperGaussian Model'] = (compute_AEP(superGaussian_model, all_x, all_y, wt_x, wt_y, type=types),
#                                                     percent_change_(superGaussian_model, all_x, all_y, wt_x, wt_y, type=types) 
#                                                     )
#         ## Uncomment to print the string representation
#         # for key in deficit_models_AEP.keys():
#         #     a, b = deficit_models_AEP[key]
#         #     down, up = a
#         #     print('\033[92m%s: '%key)
#         #     print("\t \033[0mDownstream AEP: %f (Gwh)"%down.values)
#         #     print("\t Upstream AEP: %f (Gwh)"%up.values)
#         #     print('\t Changes between upstream and downstream: %f%%'%b.values)

#         return deficit_models_AEP, models
#     return models

def n_turbine_AEP_Plot(site1, wt_x=None, wt_y=None, figsave=False,
                       cd_start=0.5, cd_end=10.0, cd_step=0.5):
    """
    Plot AEP % loss vs NUMBER OF UPSTREAM TURBINES.
    For each capacity density step, build the full upstream layout (no slicing),
    compute AEP % change on the downstream farm, and record a single point at x = N_turbines.
    Saves layout images only for cd in {2,3,4,5,6} when figsave=True.
    """
    site = site1
    if wt_x is None or wt_y is None:
        wt_x, wt_y = site.initial_position.T
    else:
        wt_x = np.array(wt_x, dtype=float)
        wt_y = np.array(wt_y, dtype=float)

    windTurbines = WindTurbines.from_WindTurbine_lst([SWT_23_93(), SWT_23_82()])
    windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]

    # Wake models
    noj_model          = noj_WF_model(site, windTurbines)
    turboNoj_model     = turboNoj_WF_model(site, windTurbines)
    bastankhah_model   = bastankhah_WF_model(site, windTurbines)
    nygaard_model      = nygaard_WF_model(site, windTurbines)
    carbajo_model      = carbajo_WF_model(site, windTurbines)
    zong_model         = zong_WF_model(site, windTurbines)
    niayifar_model     = niayifar_WF_model(site, windTurbines)
    superGaussian_model= blondelSuperGaussian_WF_model(site, windTurbines)

    # Upstream boundary (Nysted)
    filepath = "DigitizeLayers/Europe/nysted.geojson"
    eastings, northings = get_only_boundary(filepath)
    eastings, northings = np.array(eastings), np.array(northings)
    boundary_points = list(zip(eastings, northings))

    # --- set up a 2x3 layout grid for CD = 2..7 (inclusive) ---
    cd_panels = [2, 3, 4, 5, 6, 7]
    fig_layout, axes_layout = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes_layout = axes_layout.ravel()
    ax_idx = {cd: i for i, cd in enumerate(cd_panels)}

    # common limits for all subplots (from boundary)
    xmin, xmax = eastings.min(), eastings.max()
    ymin, ymax = northings.min(), northings.max()
    dx, dy = xmax - xmin, ymax - ymin
    padx, pady = 0.05 * dx, 0.05 * dy

    # Sweep capacity density (we'll store results keyed by N)
    capacity_densities = np.round(np.arange(cd_start, cd_end + 1e-9, cd_step), 2)
    results_by_N = {"N": [],
                    "NOJ": [], "TurboNOJ": [], "Bastankhah": [], "Zong": [],
                    "Niayifar": [], "SuperGaussian": [], "Carbajo": [], "Nygaard": []}

    area_km2 = 22.59
    rated_MW = 2.3
    spacing_D0 = 15.0

    for cd in tqdm(capacity_densities, desc="Capacity density steps [MW/km²]"):
        # derive N from capacity density (no slicing later)
        n_turbs = compute_number_of_turbines(area=area_km2, rated_power=rated_MW, capacity_density=cd)

        # build full upstream layout for this cd
        spacing = spacing_D0
        up_x, up_y = grid_WTposition_generator(boundary_points, SWT_23_82(), n_turbs, spacing=spacing)
        while len(up_x) != n_turbs:
            spacing -= 0.05
            up_x, up_y = grid_WTposition_generator(boundary_points, SWT_23_82(), n_turbs, spacing=spacing)

        # populate the 2x3 subplot ONLY for cd in [2..7]
        if cd in cd_panels:
            ax = axes_layout[ax_idx[cd]]
            boundary_plot(eastings, northings)
            ax.plot(up_x, up_y, '2k')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymin - pady, ymax + pady)
            ax.tick_params(axis='both', labelsize=axis_tick_size)
            ax.set_title(f"CD={cd} MW/km² | N={n_turbs}", fontsize=label_fontsize)
            # show axis labels on left/bottom panels only (cleaner)
            if ax_idx[cd] % 3 == 0:
                ax.set_ylabel("Y-UTM [m]", fontsize=label_fontsize)
            if ax_idx[cd] // 3 == 1:
                ax.set_xlabel("X-UTM [m]", fontsize=label_fontsize)

        # combine farms and compute AEP % change on downstream only
        all_x = np.r_[wt_x, up_x] if wt_x is not None else np.r_[site.initial_position.T[0], up_x]
        all_y = np.r_[wt_y, up_y] if wt_y is not None else np.r_[site.initial_position.T[1], up_y]
        types = [0] * (len(wt_x) if wt_x is not None else len(site.initial_position)) + [1] * len(up_x)

        results_by_N["N"].append(n_turbs)
        results_by_N["NOJ"].append(           percent_change_(noj_model,           all_x, all_y, wt_x, wt_y, types) )
        results_by_N["TurboNOJ"].append(      percent_change_(turboNoj_model,      all_x, all_y, wt_x, wt_y, types) )
        results_by_N["Bastankhah"].append(    percent_change_(bastankhah_model,    all_x, all_y, wt_x, wt_y, types) )
        results_by_N["Zong"].append(          percent_change_(zong_model,          all_x, all_y, wt_x, wt_y, types) )
        results_by_N["Niayifar"].append(      percent_change_(niayifar_model,      all_x, all_y, wt_x, wt_y, types) )
        results_by_N["SuperGaussian"].append( percent_change_(superGaussian_model, all_x, all_y, wt_x, wt_y, types) )
        results_by_N["Carbajo"].append(       percent_change_(carbajo_model,       all_x, all_y, wt_x, wt_y, types) )
        results_by_N["Nygaard"].append(       percent_change_(nygaard_model,       all_x, all_y, wt_x, wt_y, types) )

    # finalize the 2x3 layout grid figure
    fig_layout.suptitle("Nysted layouts", fontsize=title_fontsize)
    fig_layout.tight_layout()
    if figsave:
        fig_layout.savefig(f"Ny_ro_layout_grid_{_ID_}.png", dpi=600)

    # ----- main plot: AEP % loss vs NUMBER OF TURBINES -----
    order = np.argsort(results_by_N["N"])
    Ns = np.array(results_by_N["N"])[order]
    def _s(m): return np.array(results_by_N[m])[order]

    plt.figure(figsize=(12, 6))
    plt.plot(Ns, _s("NOJ"),            label="NOJ")
    plt.plot(Ns, _s("Bastankhah"),     label="Bastankhah")
    plt.plot(Ns, _s("Niayifar"),       label="Niayifar")
    plt.plot(Ns, _s("TurboNOJ"),       label="Turbo NOJ")
    plt.plot(Ns, _s("Zong"),           label="Zong")
    plt.plot(Ns, _s("SuperGaussian"),  label="Super Gaussian")
    plt.plot(Ns, _s("Carbajo"),        label="Carbajo")
    plt.plot(Ns, _s("Nygaard"),        label="Nygaard")

    plt.legend(fontsize=legend_fontsize)
    plt.ylabel("AEP Percent Difference [%]", fontsize=label_fontsize)
    plt.xlabel("Number of Upstream Turbines", fontsize=label_fontsize)
    plt.title("Upstream turbine count impact on Rodsand II AEP", fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=axis_tick_size)
    if figsave:
        plt.savefig(f"Ny_ro_vs_N_{_ID_}.png", dpi=600)

# In main(), keep call name the same:
def main():
    site = Rodsand_2()
    n_turbine_AEP_Plot(site, figsave=True, cd_start=0.5, cd_end=10.0, cd_step=0.5)
    print("Done")

if __name__ == "__main__":
    main()