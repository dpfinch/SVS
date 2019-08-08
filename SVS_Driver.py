### ============================================================================
### Main driver for SatLOS
### ============================================================================
from SatLOS.InputFile import *
from SatLOS import Satellite_LOS
from datetime import datetime
import pandas as pd

def Get_Input_Variables(**kwargs):
    inputs = {}
    if 'Ground_Station_Lat' in kwargs.keys():
        inputs['GrndLat'] = kwargs['Ground_Station_Lat']
    else:
        inputs['GrndLat'] = Ground_Station_Lat

    if 'Ground_Station_Lon' in kwargs.keys():
        inputs['GrndLon'] = kwargs['Ground_Station_Lon']
    else:
        inputs['GrndLon'] = Ground_Station_Lon

    if 'Ground_Station_Lat' in kwargs.keys():
        inputs['GrndAlt'] = kwargs['Ground_Station_Alt']
    else:
        inputs['GrndAlt'] = Ground_Station_Alt

    if 'Satellite_traj' in kwargs.keys():
        inputs['SatelliteTraj'] = kwargs['Satellite_traj']
    else:
        inputs['SatelliteTraj'] = Satellite_Traj

    if 'Satellite_traj_file' in kwargs.keys():
        inputs['SatelliteTrajFile'] = kwargs['Satellite_traj_file']
    else:
        inputs['SatelliteTrajFile'] = Satellite_Traj_File

    if 'Met_file_path' in kwargs.keys():
        inputs['met_file'] = kwargs['Met_file_path']
    else:
        inputs['met_file'] = Met_file_path

    return inputs

def run_LOS(**kwargs):
    inputs = Get_Input_Variables(**kwargs)

    grnd = Satellite_LOS.GroundStation(inputs['GrndLat'],inputs['GrndLon'],
                                       inputs['GrndAlt'])
    
    sat_path = pd.read_csv(inputs['SatelliteTrajFile'], index_col = 0, parse_dates = True,
                           header = None, names = ['Longitude','Latitude','Altitude'])

    all_LOSs = {}

    for sat_timestamp in sat_path.index:
        coord = sat_path.loc[sat_timestamp]
    
        satellite = Satellite_LOS.SatelliteOrbit(coord.Latitude,
                                                 coord.Longitude,
                                                 coord.Altitude,
                                                 sat_timestamp)
        az,el = Satellite_LOS.GetObserverView(grnd,satellite)
        if el < 30:
            continue
    
        hgt_layers = Satellite_LOS.GetModelLayerHeights(grnd, inputs['met_file'])
        
        layer_df = Satellite_LOS.ModelLayerCoordList(grnd,az,el,hgt_layers)
        index_df = Satellite_LOS.FindIntersects(layer_df)

        all_LOSs[sat_timestamp] = index_df

    return all_LOSs

def run_extract(LOS_coords, **kwargs):
    print('Extracting data')
    
### ============================================================================
### END OF PROGRAM
### ============================================================================
