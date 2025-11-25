import py_wake
import numpy as np
import matplotlib.pyplot as plt
from py_wake.wind_turbines import WindTurbines
import sys
import os
import random
from tqdm import tqdm
from aep_percent_diff_Plot import *
from windFarms_windTurbines import *
import pandas as pd

# Adjust the path to where your Examples directory is located
from windFarms_windTurbines import *
from cluster_analysis import *

_ID_ = random.randint(0, 1000)
title_fontsize = 20
label_fontsize = 18
legend_fontsize = 16
axis_tick_size = 14


class test_turbine(GenericWindTurbine):
    def __init__(self, D = 82.4, H = 69, P = 2300):
        """
        Parameters
        ----------
        D: Diameter (m)
        H: Hub Height (m)
        P: Rated Power (KW)
        """
        GenericWindTurbine.__init__(self, name='Test Turbine', diameter=D, hub_height=H,
                             power_norm=P, turbulence_intensity=0.07)



def rotorDiameter_AEP_Plot(site, wt_x, wt_y, neighbour_x, neighbour_y, 
                        initial_D = 30, 
                        final_D = 250, 
                        step = 10,
                        figsave = False):
    
    site1 = site
    all_x = np.r_[wt_x, neighbour_x]
    all_y = np.r_[wt_y, neighbour_y]
    types = [0]*len(wt_x) + [1]*len(neighbour_x)


    rotor_diameters = range(initial_D, final_D, step)

    deficit_models_AEP = {}

    deficit_models_AEP['NOJ Model'] = []
            
    deficit_models_AEP['TurboNOJ Model'] = []   
                                                         
    deficit_models_AEP['Bastankhah Model'] = []
            
    deficit_models_AEP['Zong Model'] = []

    deficit_models_AEP['Niayifar Model'] = []
            
    deficit_models_AEP['SuperGaussian Model'] = []

    deficit_models_AEP['Nygaard Model'] = []

    deficit_models_AEP['Carbajo Model'] = []

    for D in tqdm(rotor_diameters, desc="Processing Rotor Diameter"):
        w_turbine = test_turbine(D = D)
        windTurbines = WindTurbines.from_WindTurbine_lst([SWT_23_93(), w_turbine])
        windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]


        noj_model = noj_WF_model(site, windTurbines)
        turboNoj_model = turboNoj_WF_model(site, windTurbines)
        bastankhah_model = bastankhah_WF_model(site, windTurbines)
        nygaard_model = nygaard_WF_model(site, windTurbines)
        carbajo_model = carbajo_WF_model(site, windTurbines)
        zong_model = zong_WF_model(site, windTurbines)
        niayifar_model = niayifar_WF_model(site, windTurbines)
        superGaussian_model = blondelSuperGaussian_WF_model(site, windTurbines)


        noj_diff = percent_change_(noj_model, all_x, all_y, wt_x, wt_y, types)
        turboNoj_diff = percent_change_(turboNoj_model, all_x, all_y, wt_x, wt_y, types)
        bastankhah_diff = percent_change_(bastankhah_model, all_x, all_y, wt_x, wt_y, types)
        zong_diff = percent_change_(zong_model, all_x, all_y, wt_x, wt_y, types)
        niayifar_diff = percent_change_(niayifar_model, all_x, all_y, wt_x, wt_y, types)
        superGaussian_diff = percent_change_(superGaussian_model, all_x, all_y, wt_x, wt_y, types)
        carbajo_diff = percent_change_(carbajo_model, all_x, all_y, wt_x, wt_y, types)
        nygaard_diff = percent_change_(nygaard_model, all_x, all_y, wt_x, wt_y, types)

        deficit_models_AEP["NOJ Model"].append(noj_diff)
        deficit_models_AEP["TurboNOJ Model"].append(turboNoj_diff)
        deficit_models_AEP["Bastankhah Model"].append(bastankhah_diff)
        deficit_models_AEP["Zong Model"].append(zong_diff)
        deficit_models_AEP["Niayifar Model"].append(niayifar_diff)
        deficit_models_AEP["SuperGaussian Model"].append(superGaussian_diff)        
        deficit_models_AEP["Carbajo Model"].append(carbajo_diff)
        deficit_models_AEP["Nygaard Model"].append(nygaard_diff)        

    

    
   

    ### Algorithm to plot the wake



    plt.figure(figsize=[12, 10])
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['NOJ Model'], label="NOJ")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['Bastankhah Model'], label="Bastankhah")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['Niayifar Model'], label="Niayifar")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['TurboNOJ Model'], label="Turbo NOJ")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['Zong Model'], label="Zong")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['SuperGaussian Model'], label="Super Gaussian")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['Carbajo Model'], label="Carbajo")
    plt.plot(np.array(rotor_diameters), deficit_models_AEP['Nygaard Model'], label="Nygaard")
    plt.legend(fontsize=legend_fontsize)
    plt.ylabel("AEP Percent Difference [%]", fontsize=label_fontsize)
    plt.xlabel("Turbines's Rotor Diameter [m]", fontsize=label_fontsize)
    plt.title("Rotor Diameter's Impacts on Downstream AEP", fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=axis_tick_size)
    plt.tight_layout()
    if figsave:
        plt.savefig(f"rotorDiameter_AEP_diff_Ro&Ny_{_ID_}.png", dpi=1000)
    else:
        plt.show()





# plot AEP Percent difference vs the hub height of the turbines - Attention! Ask professor about if I shall get different values of the GWC
# based on the height of the turbine as well? Will that change the constant value of AEP?
def hubHeight_AEP_Plot(site, wt_x, wt_y, neighbour_x, neighbour_y, 
                        initial_hub_height = 30, 
                        final_hub_height = 250, 
                        step = 10,
                        figsave = False):
    
    site1 = site
    all_x = np.r_[wt_x, neighbour_x]
    all_y = np.r_[wt_y, neighbour_y]
    types = [0]*len(wt_x) + [1]*len(neighbour_x)


    hub_heights = range(initial_hub_height, final_hub_height, step)

    deficit_models_AEP = {}

    deficit_models_AEP['NOJ Model'] = []
            
    deficit_models_AEP['TurboNOJ Model'] = []   
                                             
    deficit_models_AEP['Fuga Model'] = []
            
    deficit_models_AEP['Bastankhah Model'] = []
            
    deficit_models_AEP['Zong Model'] = []

    deficit_models_AEP['Niayifar Model'] = []
            
    deficit_models_AEP['TurboGaussian Model'] = []
            
    deficit_models_AEP['SuperGaussian Model'] = []

    for height in tqdm(hub_heights, desc="Processing Hub Height"):
        w_turbine = test_turbine(H = height)
        windTurbines = WindTurbines.from_WindTurbine_lst([Haliade_X(), w_turbine])
        windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]


        # noj_model = noj_WF_model(site1, windTurbines)
        # turboNoj_model = turboNoj_WF_model(site1, windTurbines)
        # fuga_model = fuga_WF_model(site1, windTurbines)                                 
        bastankhah_model = bastankhah_WF_model(site1, windTurbines)
        # zong_model = zong_WF_model(site1, windTurbines)
        # niayifar_model = niayifar_WF_model(site1, windTurbines)
        # turboGaussian_model = turboGaussian_WF_model(site1, windTurbines)
        # superGaussian_model = blondelSuperGaussian_WF_model(site1, windTurbines)


        # noj_diff = percent_change_(noj_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # turboNoj_diff = percent_change_(turboNoj_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # fuga_diff = percent_change_(fuga_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        bastankhah_diff = percent_change_(bastankhah_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # zong_diff = percent_change_(zong_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # niayifar_diff = percent_change_(niayifar_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # turboGaussian_diff = percent_change_(turboGaussian_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)
        # superGaussian_diff = percent_change_(superGaussian_model, all_x, all_y, wt_x, wt_y, neighbour_x, neighbour_y, types)

        # deficit_models_AEP["NOJ Model"].append(noj_diff)
        # deficit_models_AEP["TurboNOJ Model"].append(turboNoj_diff)
        # deficit_models_AEP["Fuga Model"].append(fuga_diff)
        deficit_models_AEP["Bastankhah Model"].append(bastankhah_diff)
        # deficit_models_AEP["Zong Model"].append(zong_diff)
        # deficit_models_AEP["Niayifar Model"].append(niayifar_diff)
        # deficit_models_AEP["TurboGaussian Model"].append(turboGaussian_diff)
        # deficit_models_AEP["SuperGaussian Model"].append(superGaussian_diff)        


    
   
    df = pd.DataFrame(deficit_models_AEP)
    print(df)

    ### Algorithm to plot the wake



    plt.figure(figsize=[12, 10])
    # plt.plot(np.array(hub_heights), deficit_models_AEP['NOJ Model'], label="NOJ")
    plt.plot(np.array(hub_heights), deficit_models_AEP['Bastankhah Model'], label="Bastankhah")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['Fuga Model'], label="Fuga")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['Niayifar Model'], label="Niayifar")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['TurboNOJ Model'], label="Turbo NOJ")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['Zong Model'], label="Zong")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['TurboGaussian Model'], label="Turbo Gaussian")
    # plt.plot(np.array(hub_heights), deficit_models_AEP['SuperGaussian Model'], label="Super Gaussian")
    plt.legend()
    plt.ylabel("AEP Percent Difference [%]")
    plt.xlabel("Turbines's Hub height [m]")
    plt.title("Hub Height's Impacts on Downstream AEP: Revolution–South Fork vs. Vineyard Wind")
    if figsave:
        plt.savefig("hubHeight_AEP_diff_Vin&Rev")
    else:
        plt.show()





# plot AEP Percent difference vs the rated power of the turbines
def ratedPower_AEP_Plot(site, wt_x, wt_y, neighbour_x, neighbour_y, 
                        initial_rated_power = 500, 
                        final_rated_power = 20000, 
                        step = 250,
                        figsave = False):
    
    site1 = site
    all_x = np.r_[wt_x, neighbour_x]
    all_y = np.r_[wt_y, neighbour_y]
    types = [0]*len(wt_x) + [1]*len(neighbour_x)


    rated_power = range(initial_rated_power, final_rated_power, step)

    deficit_models_AEP = {}

    deficit_models_AEP['NOJ Model'] = []
            
    deficit_models_AEP['TurboNOJ Model'] = []   
                                                         
    deficit_models_AEP['Bastankhah Model'] = []
            
    deficit_models_AEP['Zong Model'] = []

    deficit_models_AEP['Niayifar Model'] = []
            
    deficit_models_AEP['SuperGaussian Model'] = []

    deficit_models_AEP['Nygaard Model'] = []

    deficit_models_AEP['Carbajo Model'] = []


    for power in tqdm(rated_power, desc="Processing Rated Power"):
        w_turbine = test_turbine(P = power)
        windTurbines = WindTurbines.from_WindTurbine_lst([SWT_23_93(), w_turbine])
        windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]


        noj_model = noj_WF_model(site, windTurbines)
        turboNoj_model = turboNoj_WF_model(site, windTurbines)
        bastankhah_model = bastankhah_WF_model(site, windTurbines)
        nygaard_model = nygaard_WF_model(site, windTurbines)
        carbajo_model = carbajo_WF_model(site, windTurbines)
        zong_model = zong_WF_model(site, windTurbines)
        niayifar_model = niayifar_WF_model(site, windTurbines)
        superGaussian_model = blondelSuperGaussian_WF_model(site, windTurbines)


        noj_diff = percent_change_(noj_model, all_x, all_y, wt_x, wt_y, types)
        turboNoj_diff = percent_change_(turboNoj_model, all_x, all_y, wt_x, wt_y, types)
        bastankhah_diff = percent_change_(bastankhah_model, all_x, all_y, wt_x, wt_y, types)
        zong_diff = percent_change_(zong_model, all_x, all_y, wt_x, wt_y, types)
        niayifar_diff = percent_change_(niayifar_model, all_x, all_y, wt_x, wt_y, types)
        superGaussian_diff = percent_change_(superGaussian_model, all_x, all_y, wt_x, wt_y, types)
        carbajo_diff = percent_change_(carbajo_model, all_x, all_y, wt_x, wt_y, types)
        nygaard_diff = percent_change_(nygaard_model, all_x, all_y, wt_x, wt_y, types)

        deficit_models_AEP["NOJ Model"].append(noj_diff)
        deficit_models_AEP["TurboNOJ Model"].append(turboNoj_diff)
        deficit_models_AEP["Bastankhah Model"].append(bastankhah_diff)
        deficit_models_AEP["Zong Model"].append(zong_diff)
        deficit_models_AEP["Niayifar Model"].append(niayifar_diff)
        deficit_models_AEP["SuperGaussian Model"].append(superGaussian_diff)        
        deficit_models_AEP["Carbajo Model"].append(carbajo_diff)
        deficit_models_AEP["Nygaard Model"].append(nygaard_diff)      


    
   

    ### Algorithm to plot the wake



    plt.figure(figsize=[12, 10])
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['NOJ Model'], label="NOJ")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['Bastankhah Model'], label="Bastankhah")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['Niayifar Model'], label="Niayifar")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['TurboNOJ Model'], label="Turbo NOJ")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['Zong Model'], label="Zong")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['SuperGaussian Model'], label="Super Gaussian")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['Carbajo Model'], label="Carbajo")
    plt.plot(np.array(rated_power) / 1000, deficit_models_AEP['Nygaard Model'], label="Nygaard")

    plt.legend(fontsize=legend_fontsize)
    plt.ylabel("AEP Percent Difference [%]", fontsize=label_fontsize)
    plt.xlabel("Rated Power [MW]", fontsize=label_fontsize)
    plt.title("Rated Power Impacts on Downstream AEP", fontsize=title_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=axis_tick_size)
    plt.tight_layout()
    if figsave:
        plt.savefig(f"RatedPower_AEP_diff_Ro&Ny_{_ID_}.png", dpi=1000)
    else:
        plt.show()

def main():
    site = Rodsand_2()
    wt_x, wt_y = np.load("wind_turbine_layouts/rodsand2_layout.npy")
    neighbour_x_i, neighbour_y_i = np.load("wind_turbine_layouts/nysted_layout.npy")
    neighbour_x = neighbour_x_i # 3 km apart (Euclidean Distance)
    neighbour_y = neighbour_y_i
    print("Process Rated Power")
    ratedPower_AEP_Plot(site, wt_x, wt_y, neighbour_x, neighbour_y, figsave=True)
    print("Rated Power Done")


    print("Process Rotor Diameter")
    rotorDiameter_AEP_Plot(site, wt_x, wt_y, neighbour_x, neighbour_y, figsave=True)
    print("Rotor Diameter Done")

if __name__=="__main__":
    main()