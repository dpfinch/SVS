### ============================================================================
### Set the parameters for the SVS module and run the main program
### ============================================================================
import SVS
from datetime import datetime, timedelta
import pandas as pd
import netCDF4 as nc4
### ============================================================================

def run_sps(G5NR_file_path):

    sat_az_el_file = '/Users/dfinch/Documents/svs_20190904_171119.csv'
    print('Extracting azimuth and elevation angles from {}'.format(sat_az_el_file))
    print(' ')

    ogs_coords = {}
    with open(sat_az_el_file, 'r') as txt_file:
        for i, line in enumerate(txt_file):
            if i == 2:
                latitude = float(line.split(':')[-1])
            if i == 3:
                longitude = float(line.split(':')[-1])
            if i == 4:
                altitude = float(line.split(':')[-1]) /1000 # Convert m to km

    
    grnd_stn = SVS.GroundStation(latitude,longitude, altitude)
    
    overpasses = pd.read_csv(sat_az_el_file, skiprows = 5, index_col = 'Time',
                             parse_dates = True, skipinitialspace = True)

    pass_LOS = {}
    print('Calculating line of sight intersects...')
    print(' ')
    for index, row in overpasses.iterrows():
        hgt_layers = SVS.GetModelLayerHeights(grnd_stn, G5NR_file_path, row.name.to_pydatetime())
        layer_df = SVS.ModelLayerCoordList(grnd_stn,row.Azimuth,row.Elevation,hgt_layers)
        pass_LOS[index] = SVS.FindIntersects(layer_df)
    return pass_LOS


def Extract_G5NR_Data(index_dict,G5NR_file_path):
    ## File names and variables:
    ##
    ## Cloud:
    ## inst30mn_3d_QV_Nv: Water vapor
    ## inst30mn_3d_QI_Nv: Cloud ice condensate
    ## inst30mn_3d_QL_Nv: Cloud water condensate
    ## inst30mn_3d_QR_Nv: Falling Rain
    ## inst30mn_3d_QS_Nv: Falling Snow
    ##
    ## Aerosol:
    ## inst30mn_3d_BCPHILIC_Nv: Hydrophilic Black Carbon
    ## inst30mn_3d_BCPHOBIC_Nv: Hydrophobic Black Carbon
    ## inst30mn_3d_DU001_Nv: Dust Bin 1 (0.7-1 microns)
    ## inst30mn_3d_DU002_Nv: Dust Bin 2 (1-1.8 microns)
    ## inst30mn_3d_DU003_Nv: Dust Bin 3 (1.8-3 microns)
    ## inst30mn_3d_DU004_Nv: Dust Bin 4 (3-6 microns)
    ## inst30mn_3d_DU005_Nv: Dust Bin 5 (6-10 microns)
    ## inst30mn_3d_OCPHILIC_Nv: Hydrophilic Organic Carbon
    ## inst30mn_3d_OCPHOBIC_Nv: Hydrophobic Organic Carbon
    ## inst30mn_3d_SS001_Nv: Sea Salt Bin 1 (0.03-0.1 microns)
    ## inst30mn_3d_SS002_Nv: Sea Salt Bin 2 (0.1-0.5 microns)
    ## inst30mn_3d_SS003_Nv: Sea Salt Bin 3 (0.5-1.5 microns)
    ## inst30mn_3d_SS004_Nv: Sea Salt Bin 4 (1.5-5 microns)
    ## inst30mn_3d_SS005_Nv: Sea Salt Bin 5 (5-10 microns)

    atmos_vars = [
    'BCPHILIC','BCPHOBIC','DU001','DU002','DU003','DU004','DU005','OCPHILIC',
    'OCPHOBIC','QI','QL','QR','QS','QV','SS001','SS002','SS003','SS004',
    'SS005']
    # Only use following variables since they're the only ones on the harddrive currently
    atmos_vars = ['QL','QI','QV', 'BCPHILIC']

    all_vars_df = []
    for variable in atmos_vars:
        print('Extracting data for {}.'.format(variable))
        datetime_holder = None

        overpass_time = []
        model_data_sum = []

        for k in index_dict.keys():
            dt = k.to_pydatetime()

            if dt != datetime_holder:
                open_dataset = SVS.ImportData(G5NR_file_path,variable,dt)
                datetime_holder = dt

            pass_df = index_dict[k]
            LOS_data = []
            for n, r in pass_df.iterrows():
                LOS_data.append(float(open_dataset.variables[variable][0,r.Layer,r.LatIndex,r.LonIndex]))

            overpass_time.append(dt)
            model_data_sum.append(sum(LOS_data))

        all_vars_df.append(pd.DataFrame({variable:model_data_sum},index = overpass_time))

    data_df = pd.concat(all_vars_df, axis = 1)
    return data_df

def Plot_Paths(index_dict,G5NR_file_path):
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    grnd = SVS.GroundStation(51.14456,-1.43858,0.01)
    hgt_layers = SVS.GetModelLayerHeights(grnd,G5NR_file_path, datetime(2006,4,3,6,9,0))
    alt_layers = np.cumsum(hgt_layers[:-1,0,0])

    for k in index_dict.keys():
        if k.hour == 6:
            continue
        temp = index_dict[k]
        
        alts = []
        for l in temp.Layer:
            alts.append(alt_layers[l])
                
        ax.scatter3D(temp.LonIndex,temp.LatIndex,alts)

    plt.show()

if __name__ == '__main__':

    G5NR_file_path = '/Volumes/DougsDrive/'
    LOS = run_sps(G5NR_file_path)
    model_data = Extract_G5NR_Data(LOS,G5NR_file_path)
    print('OUTPUT:')
    print(model_data)

### ============================================================================
### END OF PROGRAM
### ============================================================================
