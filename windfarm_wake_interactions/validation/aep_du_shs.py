
import py_wake
import numpy as np
import matplotlib.pyplot as plt
from py_wake.wind_turbines import WindTurbines
import sys
import os
from tqdm import tqdm

# Adjust the path to where your Examples directory is located
from windFarms_windTurbines import *
from aep_percent_diff_Plot import *
from layout_dev import *
from cluster_analysis import *

_ID_ = random.randint(0, 1000)
title_fontsize = 20
label_fontsize = 18
legend_fontsize = 16
axis_tick_size = 14

def main():
    # geojson_files = {
    # 'skip_jack_wind.geojson': (Haliade_X_12(), 10, False, True),
    # 'delaware_lease_area.geojson': (Haliade_X_12(), compute_number_of_turbines(area=282.87, rated_power=12), False, False),
    # 'us_wind.geojson': (HC_Moura(), 114, False, False),
    # }
    site = Dudgeon()

    # boundary_points = get_boundary_in_utm("test_geometry/skip_jack_wind.geojson")
    # Du_wt_x, Du_wt_y = random_WTposition_generator(10, boundary_points, Haliade_X_12(), spacing=9)
    Du_wt_x, Du_wt_y = np.load("wind_turbine_layouts/dudgeon_layout.npy")
    shs_wt_x, shs_wt_y = np.load("wind_turbine_layouts/sheringham_shoal_layout.npy")
    
    
    point_dist_x = np.array([3.784e05, 3.8795e05])
    point_dist_y = np.array([5.89031e06, 5.90267e06])

    all_x = np.r_[Du_wt_x, shs_wt_x]
    all_y = np.r_[Du_wt_y, shs_wt_y]
    # plt.figure(figsize=[10, 8])
    # # for key in geojson_files.keys():    
    # #         boundary_points = None
    # #         # filepath = directory + key
    # #         # eastings, northings = get_only_boundary(filepath)
    # #         # eastings = np.array(eastings)
    # #         # northings = np.array(northings)
    # #         boundary_plot(eastings, northings)
    # # plt.scatter(all_x, all_y, c=['k*']*len(Du_wt_x) + ['r*']*len(shs_wt_x), s=100, label='Wind Turbines', edgecolors='k')
    # plt.plot(point_dist_x, point_dist_y, '--', ms=5, color='r')
    # plt.plot(all_x, all_y, '2k', ms=3.75)
    # plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.title("Layout Setup", fontsize=18)
    # plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot aspect ratio is equal
    # plt.xlabel("X-UTM Coordinates [m]", fontsize=15)
    # plt.ylabel("Y-UTM Coordinates [m]", fontsize=15)
    # plt.xlim(505000, 560000)
    # plt.axis([505000, 560000, 4.225e06, 4.295e06])
    # plt.show()
    # exit(0)
    windfarm_distance = []
    # percent_diff = []

    windTurbines = WindTurbines.from_WindTurbine_lst([SWT_60_154(), SWT_36_107()])
    windTurbines._names = ["Current Wind Farm", "Neighbour Wind Farm"]

    deficit_models_AEP = {}

    deficit_models_AEP['NOJ Model'] = []
            
    deficit_models_AEP['TurboNOJ Model'] = []   
                                                         
    deficit_models_AEP['Bastankhah Model'] = []
            
    deficit_models_AEP['Zong Model'] = []

    deficit_models_AEP['Niayifar Model'] = []
                        
    deficit_models_AEP['SuperGaussian Model'] = []

    deficit_models_AEP['Nygaard Model'] = []

    deficit_models_AEP['Carbajo Model'] = []

    noj_model = noj_WF_model(site, windTurbines)
    turboNoj_model = turboNoj_WF_model(site, windTurbines)
    bastankhah_model = bastankhah_WF_model(site, windTurbines)
    zong_model = zong_WF_model(site, windTurbines)
    niayifar_model = niayifar_WF_model(site, windTurbines)
    superGaussian_model = blondelSuperGaussian_WF_model(site, windTurbines)
    nygaard_model = nygaard_WF_model(site, windTurbines)
    carbajo_model = carbajo_WF_model(site, windTurbines)

    
    models = np.array([noj_model, 
                       turboNoj_model, 
                       bastankhah_model, 
                  zong_model, 
                  niayifar_model, 
                  superGaussian_model,
                  nygaard_model,
                  carbajo_model
                  ])

    # Define your range of shifts
    #-15000
    # 100000
    shift_values = range(-5000, 50000, 2500)

    # The Starting euclidean distance d between the 2 sites is aprox 9 km (wt = 125 and wt = 44)
    # the ending d is 70 km
    base_x = point_dist_x[0]
    base_y = point_dist_y[0]
    # print(x)
    # print(y)
    a, b = 0, 0
    all_x = []
    all_y = []
    skip = False
    for wf_model, key in zip(models, deficit_models_AEP.keys()):
        # Wrap shift_values in tqdm for the progress bar
        print(f'Model Processing: {key}')
        for shift in tqdm(shift_values, desc="Processing Distances"):
            #print(shift)
            neighbour_x = shs_wt_x - shift
            neighbour_y = shs_wt_y - shift
            pos_x = base_x - shift
            pos_y = base_y - shift
            # print(pos_x)
            # print(pos_y)

            all_x = np.r_[Du_wt_x, neighbour_x]
            all_y = np.r_[Du_wt_y, neighbour_y]
            types = [0]*len(Du_wt_x) + [1]*len(neighbour_x)

            # Computation of percentage difference in AEP due to wake interactions
            p_diff = percent_change_(wf_model, all_x, all_y, Du_wt_x, Du_wt_y, types)
            deficit_models_AEP[key].append(p_diff)
            a = pos_x
            b = pos_y
            pos_x_dist = point_dist_x[1] - pos_x
            pos_y_dist = point_dist_y[1] - pos_y
            wt_distance = ((pos_x_dist**2) + (pos_y_dist**2))**0.5
            if not skip:
                windfarm_distance.append(wt_distance/1000) #km
            # print("Distance Between the Sites: " + str(wt_distance) + " m")
            # break
        skip=True

    # plt.figure(figsize=[12, 10])
    # plt.plot([a, point_dist_x[1]], [b, point_dist_y[1]], '-.')
    # plt.plot(all_x, all_y, '2k')
    # plt.xlabel("X-UTM Coordinates [m]", fontsize=label_fontsize)
    # plt.ylabel("Y-UTM Coordinates [m]", fontsize=label_fontsize)
    # plt.show()
    # exit(0)
    # Plotting the results  
    plt.figure(figsize=[12, 10])
    plt.plot(windfarm_distance, deficit_models_AEP['NOJ Model'], label="NOJ")
    plt.plot(windfarm_distance, deficit_models_AEP['Bastankhah Model'], label="Bastankhah")
    plt.plot(windfarm_distance, deficit_models_AEP['Niayifar Model'], label="Niayifar")
    plt.plot(windfarm_distance, deficit_models_AEP['TurboNOJ Model'], label="Turbo NOJ")
    plt.plot(windfarm_distance, deficit_models_AEP['Zong Model'], label="Zong")
    plt.plot(windfarm_distance, deficit_models_AEP['SuperGaussian Model'], label="Super Gaussian")
    plt.plot(windfarm_distance, deficit_models_AEP['Carbajo Model'], label="Carbajo")
    plt.plot(windfarm_distance, deficit_models_AEP['Nygaard Model'], label="Nygaard")

    plt.legend(fontsize=legend_fontsize)
    plt.ylabel("AEP Percent Difference [%]", fontsize=label_fontsize)
    plt.xlabel("d [km]", fontsize=label_fontsize)
    plt.title("AEP Percent Loss for Different Distances d", fontsize=title_fontsize)
    plt.tick_params(axis='both', labelsize=axis_tick_size)
    plt.tight_layout()
    plt.savefig(f"AEP_diff_Du&Shs_{_ID_}", dpi=1000)
    # plot_wf_wake_maps(wf_model, 25, 290, all_x, all_y, "Vin vs REV-Fork", 500, "Basthankha", type=types)
    plt.show()
if __name__=="__main__":

    main()