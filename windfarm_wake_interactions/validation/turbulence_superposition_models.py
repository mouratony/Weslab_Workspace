import py_wake
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from py_wake.wind_turbines import WindTurbines
import sys
import os
import random
from tqdm import tqdm
from py_wake.superposition_models import *
from py_wake.turbulence_models import *
py_wake
import pandas as pd
# Adjust the path to where your Examples directory is located
from windFarms_windTurbines import *
from cluster_analysis import *


_ID_ = random.randint(0, 1000)
title_fontsize = 16
label_fontsize = 14
legend_fontsize = 10
axis_tick_size = 10

superposition_models = {
    'LinearSum': LinearSum(),
    'SquaredSum': SquaredSum(),
    #'WeightedSum': WeightedSum(),
    'MaxSum': MaxSum()
}

turbulence_models = {
    'STF2005': STF2005TurbulenceModel(),
    'STF2017': STF2017TurbulenceModel(),
    'GCL': GCLTurbulence(),
    'CrespoHernandez': CrespoHernandez()
}


if __name__=="__main__":
    site = Dudgeon()
    wt_x, wt_y = np.load("wind_turbine_layouts/dudgeon_layout.npy")
    neighbour_x_i, neighbour_y_i = np.load("wind_turbine_layouts/sheringham_shoal_layout.npy")
    neighbour_x = neighbour_x_i
    neighbour_y = neighbour_y_i

    all_x = np.r_[wt_x, neighbour_x]
    all_y = np.r_[wt_y, neighbour_y]
    types = [0]*len(wt_x) + [1]*len(neighbour_x)

    windTurbines = WindTurbines.from_WindTurbine_lst([SWT_60_154(), SWT_36_107()])
    windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]
    

    # Superposition models with Bastankhah WF model
    n_cpu = None
    # Bastankhah_models_SP = {
    #     'BAS LinearSum() - Original': bastankhah_WF_model(site, 
    #                                                       windTurbines)(all_x, all_y, type=types, 
    #                                                                     n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS SquaredSum()': bastankhah_WF_model(site, 
    #                                             windTurbines, 
    #                                             superpositionModel=SquaredSum())(all_x, all_y, type=types, 
    #                                                                              n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS WeightedSum()': bastankhah_WF_model(site, 
    #                                              windTurbines, 
    #                                              superpositionModel=WeightedSum())(all_x, all_y, type=types, 
    #                                                                                n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     # Auto Set the superposition model to SqrMaxSum when using CrespoHernandez turbulence model
    #     'BAS SqrMaxSum()': bastankhah_WF_model(site, 
    #                                            windTurbines, 
    #                                            turbulenceModel=CrespoHernandez())(all_x, all_y, type=types, 
    #                                                                               n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS MaxSum()': bastankhah_WF_model(site, 
    #                                         windTurbines, 
    #                                         superpositionModel=MaxSum())(all_x, all_y, type=types,
    #                                                                      n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    # }


    ##########################################################################################
    ### In case you want to check the Turbulence models are being changed properly
    ##########################################################################################
    # Bastankhah_models_turb = {
    #     'BAS - Original': bastankhah_WF_model(site, 
    #                                          windTurbines),#.aep(all_x, all_y, n_cpu = n_cpu).sum(),

    #     'BAS CrespoHernandez()': bastankhah_WF_model(site, 
    #                                             windTurbines, 
    #                                             turbulenceModel=CrespoHernandez()),#.aep(all_x, all_y, n_cpu = n_cpu).sum(),

    #     'BAS STF2005TurbulenceModel()': bastankhah_WF_model(site, 
    #                                              windTurbines, 
    #                                              turbulenceModel=STF2005TurbulenceModel()),#.aep(all_x, all_y, n_cpu = n_cpu).sum(),

    #     # Auto Set the superposition model to SqrMaxSum when using CrespoHernandez turbulence model
    #     'BAS STF2017TurbulenceMode()': bastankhah_WF_model(site, 
    #                                            windTurbines, 
    #                                            turbulenceModel=STF2017TurbulenceModel()),#.aep(all_x, all_y, n_cpu = n_cpu).sum(),

    #     'BAS GCLTurbulence()': bastankhah_WF_model(site, 
    #                                         windTurbines, 
    #                                         turbulenceModel=GCLTurbulence())#.aep(all_x, all_y, n_cpu = n_cpu).sum()
    # }

    # for i in Bastankhah_models_turb.values():
    #     print(i)
    # exit(0)    
    ##########################################################################################
    ##########################################################################################


    # Bastankhah_models_turb = {
    #     'BAS - Original': bastankhah_WF_model(site, 
    #                                          windTurbines)(all_x, all_y, type=types, 
    #                                                        n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS CrespoHernandez()': bastankhah_WF_model(site, 
    #                                             windTurbines, 
    #                                             turbulenceModel=CrespoHernandez())(all_x, all_y, type=types, 
    #                                                                                n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS STF2005TurbulenceModel()': bastankhah_WF_model(site, 
    #                                              windTurbines, 
    #                                              turbulenceModel=STF2005TurbulenceModel())(all_x, all_y, type=types, 
    #                                                                                        n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     # Auto Set the superposition model to SqrMaxSum when using CrespoHernandez turbulence model
    #     'BAS STF2017TurbulenceMode()': bastankhah_WF_model(site, 
    #                                            windTurbines, 
    #                                            turbulenceModel=STF2017TurbulenceModel())(all_x, all_y, type=types, 
    #                                                                                      n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum(),

    #     'BAS GCLTurbulence()': bastankhah_WF_model(site, 
    #                                         windTurbines, 
    #                                         turbulenceModel=GCLTurbulence())(all_x, all_y, type=types, 
    #                                                                          n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    # }
    # test_model = blondelSuperGaussian_WF_model(site, 
    #                                         windTurbines, 
    #                                         turbulenceModel=GCLTurbulence())(all_x, all_y, type=types, 
    #                                                                          n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    # print(test_model)
    # exit(0)

    noj_original_model = noj_WF_model(site, windTurbines)(all_x, all_y, type=types, 
                                                   n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    
    turboNoj_original_model = turboNoj_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                         n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    
    bas_original_model = bastankhah_WF_model(site, windTurbines)(all_x, all_y, type=types, 
                                                           n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()

    zong_original_model = zong_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                   n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum() 
    
    niayifar_original_model = niayifar_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                         n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    
    carbajo_original_model = carbajo_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                       n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    
    superGaussian_orginal_model = blondelSuperGaussian_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                                    n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    Nygaard_original_model = nygaard_WF_model(site, windTurbines)(all_x, all_y, type=types,
                                                       n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()                

    print("Wakes Models Initialized.")

    Noj_models_SP = {'Original': noj_original_model} 
    TurboNoj_models_SP = {'Original': turboNoj_original_model}
    Bastankhah_models_SP = {'Original': bas_original_model}
    Zong_models_SP = {'Original': zong_original_model}
    Niayifar_models_SP = {'Original': niayifar_original_model}
    Carbajo_models_SP = {'Original': carbajo_original_model}
    SuperGaussian_models_SP = {'Original': superGaussian_orginal_model}
    Nygaard_models_SP = {'Original': Nygaard_original_model}


    Noj_models_turb = {'Original': noj_original_model}
    TurboNoj_models_turb = {'Original': turboNoj_original_model}
    Bastankhah_models_turb = {'Original': bas_original_model}
    Zong_models_turb = {'Original': zong_original_model}
    Niayifar_models_turb = {'Original': niayifar_original_model}
    Carbajo_models_turb = {'Original': carbajo_original_model}
    SuperGaussian_models_turb = {'Original': superGaussian_orginal_model}
    Nygaard_models_turb = {'Original': Nygaard_original_model}
    

    print("Calculating AEP for Different Superposition and Turbulence Models...")
    for k in superposition_models.keys():



        Noj_models_SP[k] = noj_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        
        TurboNoj_models_SP[k] = turboNoj_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Bastankhah_models_SP[k] = bastankhah_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Zong_models_SP[k] = zong_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Niayifar_models_SP[k] = niayifar_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Carbajo_models_SP[k] = carbajo_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        SuperGaussian_models_SP[k] = blondelSuperGaussian_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Nygaard_models_SP[k] = nygaard_WF_model(site, windTurbines, superpositionModel=superposition_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()

        
        
    for k in turbulence_models.keys():
        Noj_models_turb[k] = noj_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        TurboNoj_models_turb[k] = turboNoj_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Bastankhah_models_turb[k] = bastankhah_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Zong_models_turb[k] = zong_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Niayifar_models_turb[k] = niayifar_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Carbajo_models_turb[k] = carbajo_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        SuperGaussian_models_turb[k] = blondelSuperGaussian_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
        
        Nygaard_models_turb[k] = nygaard_WF_model(site, windTurbines, turbulenceModel=turbulence_models[k])(all_x, all_y, type=types, 
                                                                             n_cpu = n_cpu).aep().isel(wt=np.arange(len(wt_x))).sum()
    print("AEP Calculations Completed.")

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    axes = ax.ravel()

    # group series for each subplot
    series_sp = [
        ('Bastankhah', Bastankhah_models_SP),
        ('NOJ',        Noj_models_SP),
        ('TurboNOJ',   TurboNoj_models_SP),
        ('Zong',       Zong_models_SP),
        ('Niayifar',   Niayifar_models_SP),
        ('Carbajo',    Carbajo_models_SP),
        ('SuperGaussian', SuperGaussian_models_SP),
        ('Nygaard',    Nygaard_models_SP),
    ]
    series_ti = [
        ('Bastankhah', Bastankhah_models_turb),
        ('NOJ',        Noj_models_turb),
        ('TurboNOJ',   TurboNoj_models_turb),
        ('Zong',       Zong_models_turb),
        ('Niayifar',   Niayifar_models_turb),
        ('Carbajo',    Carbajo_models_turb),
        ('SuperGaussian', SuperGaussian_models_turb),
        ('Nygaard',    Nygaard_models_turb),
    ]

    # x categories from first dict
    cats_sp = list(series_sp[0][1].keys())
    x_sp = np.arange(len(cats_sp))
    cats_ti = list(series_ti[0][1].keys())
    x_ti = np.arange(len(cats_ti))

    m = len(series_sp)  # number of series
    width = min(1 / m, 0.12)  # keep group width under 0.8; cap per-bar width

    # helper to plot grouped bars
    def grouped_bars(ax, x, cats, series, group_pad=0.3):
        """group_pad: fraction to enlarge spacing between category centers (0.0 = no extra gap)."""
        base_x = x * (1 + group_pad)  # increase distance between groups
        bars_by_series = []
        for i, (label, data) in enumerate(series):
            offsets = base_x + (i - (m - 1) / 2) * width
            vals = [data[c] for c in cats]
            bars = ax.bar(offsets, vals, width=width, alpha=0.8, label=label)
            bars_by_series.append(bars)
        ax.set_xticks(base_x)
        ax.set_xticklabels(cats, fontsize=axis_tick_size)
        return bars_by_series

    print("Plotting Results...")
    bars_sp = grouped_bars(axes[0], x_sp, cats_sp, series_sp, group_pad=0.5)
    axes[0].set_title('AEP for Different Superposition Models', fontsize=title_fontsize)

    bars_ti = grouped_bars(axes[1], x_ti, cats_ti, series_ti, group_pad=0.5)
    axes[1].set_title('AEP for Different Turbulence Models', fontsize=title_fontsize)

    # common styling and value labels
    for ax in axes:
        ax.set_ylabel('AEP (GWh)', fontsize=label_fontsize)
        ax.yaxis.set_major_locator(MultipleLocator(250))  # denser ticks
        ax.tick_params(axis='y', labelsize=axis_tick_size)
        ax.grid(linestyle='--', alpha=0.4, axis='y')

    # annotate all bars with smaller text
    label_fs = axis_tick_size - 3
    for bars_group in (bars_sp, bars_ti):
        # flatten BarContainers
        for bars in bars_group:
            for bar in bars:
                h = bar.get_height() + 3
                ax = bar.axes
                ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(1, 3), textcoords='offset points', va='center',
                    rotation=90, rotation_mode='anchor',
                    fontsize=label_fs, clip_on=False)

    # y-limits with headroom per subplot
    for ax in axes:
        ymax = ax.get_ylim()[1]
        data = [c.get_height() for cont in ax.containers for c in cont]
        data_max = max(data)
        data_min = min(data)
        
        ax.set_ylim(min(data)*0.8, max(ymax, data_max * 1.10))

    # single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=legend_fontsize, frameon=False)
    plt.subplots_adjust(bottom=0.12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()