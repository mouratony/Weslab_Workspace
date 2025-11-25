import sys
sys.path.append('../')
from windFarms_windTurbines import *
from py_wake.wind_turbines import WindTurbines
from py_wake.flow_map import XYGrid
from cluster_analysis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from py_wake.deficit_models.no_wake import NoWakeDeficit



geojson_files = {
'baystatewind.geojson': (Haliade_X(), None, False, False), 
'beaconWind.geojson': (Haliade_X(), None, False, False),
'NewEnglandWind1.geojson': (Haliade_X(), None, False, False),
'NewEnglandWind2.geojson': (Haliade_X(), None, False, False),
'SouthCoastWind.geojson': (Haliade_X(), 141, False, False),
'Vineyard.geojson': (Haliade_X(), 62, True, VineyardWT_x, VineyardWT_y),
'vineyardNortheast.geojson': (Haliade_X(), 160, False, False),
'Revolution.geojson': (SG_110_200_DD(), 65, True, RevolutionSouthForkWT_x, RevolutionSouthForkWT_y),
'sunrise_bay_state.geojson': (SG_110_200_DD(), None, False, False),
'sunrise_wind.geojson': (SG_110_200_DD(), 84, False, True),
}


if __name__=='__main__':
    site = clusterWF_EastUS()
    turbines_models = [Haliade_X(), SG_110_200_DD()]
    windTurbines = WindTurbines.from_WindTurbine_lst(turbines_models)
    wf_model = nygaard_WF_model(site, windTurbines)

    # Plot cluster boundaries
    def plot_boundary():
        for boundary_path in geojson_files.keys():
            filepath = '../test_geometry/' + boundary_path
            eastings, northings = get_only_boundary(filepath)
            boundary_plot(eastings, northings)

    # Load turbine layout (Vineyard + Revolution etc.)
    all_x = np.load("../Data/Turbine_position_layouts/cluster_Vineyard_WT_X.npy")
    all_y = np.load("../Data/Turbine_position_layouts/cluster_Vineyard_WT_y.npy")


    buffer = 7000
    res = 500
    x_min, x_max = all_x.min()-buffer, all_x.max()+buffer
    y_min, y_max = all_y.min()-buffer, all_y.max()+buffer
    grid = XYGrid(x=np.arange(x_min, x_max, res), y=np.arange(y_min, y_max, res))

    # === A. Run WF and NWF simulations ===
    # Wake model (e.g. NOJ)
    sim_res_wake = wf_model(all_x, all_y, ws=[10])
    fm_wake = sim_res_wake.flow_map(grid)
    WS_WF_map = fm_wake.WS_eff.squeeze()

    # No-wake model (same turbines, no wake)
    no_wake_model = PropagateDownwind(site, windTurbines, wake_deficitModel=NoWakeDeficit())
    sim_res_nwf = no_wake_model(all_x, all_y, wd=[270], ws=[10])
    fm_nwf = sim_res_nwf.flow_map(grid)
    WS_NWF_map = fm_nwf.WS_eff.squeeze()

    # === B. Compute true spatial WS Deficit (%)
    ws_deficit_map = ((WS_NWF_map - WS_WF_map) / WS_NWF_map) * 100
    ws_deficit_map = np.clip(ws_deficit_map, 1, 17)

    plt.figure(figsize=(10, 8))
    c = plt.contourf(fm_wake.x, fm_wake.y, ws_deficit_map, levels=np.linspace(1, 17, 31), cmap='plasma')

    plot_boundary()
    # plt.scatter(all_x, all_y, c='k', s=5, alpha=0.6)

    # cs2 = plt.contour(fm.x, fm.y, ws_deficit_map, levels=[2, 5], colors='white', linewidths=1.5, linestyles='--')
    # plt.clabel(cs2, fmt='%1.0f%% deficit', colors='white', fontsize=9)

    cbar = plt.colorbar(c, label='Wind Speed Deficit (%)')
    cbar.set_ticks([1, 5, 9, 13, 17])

    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.axis('equal')
    plt.title('WS Deficit Map (Interpolated via XYGrid)')
    plt.tight_layout()
    plt.show()

    

