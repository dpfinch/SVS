### ============================================================================
### Extract met data from the NASA Nature Model run files and put them in a
### managable format
### ============================================================================
from netCDF4 import Dataset
import math
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pyorbital import astronomy
from pyorbital.orbital import Orbital
import geopy
from geopy.distance import distance as geodist
from skimage.draw import line
### ============================================================================
'''
1) Calculate model layer thickness and model height
2) Open viewing elevation angle and azimuth file
3) Find where the line of sight hits the top of the model grid
4) Do 3D trig to dtermine distance travelled in one model layer north and east
5) Transfer this distance to grid cells to get coord for next layer
6) Store which cells the line of sight have gone through (within one layer)
    using the Bresenham algorithm
7) Move on to next layer and repeat 4-6
8) End up with list of cells which the line  passes through. Use this to
    extract data from the model grid
'''


'''
Input of coord for ground station and satellite, and time
'''

### ============================================================================
### Tools for atmospheric height and grid calculations
### ============================================================================

def HeightDifference(temp, pres_1, pres_2):
    g0 = 9.81 # Average gravity 
    Rd = 287.0 # Gas constant for dry air
    T = temp # In kelvin
    z0 = pres_1 # Lower atmospheric layer 
    z1 = pres_2 # Upper atmospheric layer
    
    h_diff = ((Rd*T)/g0) * math.log(z0/z1)

    return h_diff

def PressureLevelEdges(pressure_diff_list):
    # Assumes list is coming from NASA Nature Run and therefore top of atmosphere
    # is level 0 and surface is level 71
    
    pres_levs = np.cumsum(pressure_diff_list)
    pres_levs_from_surf = pres_levs[::-1]

    return pres_levs_from_surf

def BoxHeights(pressure_diffs, temperatures):
    
    pres_levs = PressureLevelEdges(pressure_diffs)
    # Need to reverse temperature array to match pressure array
    temps = temperatures[::-1]
    box_heights = [HeightDifference(temps[x], pres_levs[x],pres_levs[x + 1]) for x in range(len(temps)-1)]

    return box_heights

def MeanTopOfAtmosphere(heights_grid, km = True):
    '''
        Find the mean top of the atmosphere for the given model grid
    '''
    TOA = np.mean(np.sum(heights_grid, axis =0))
    if km:
        return TOA/1000
    else:
        return TOA

def GetCoordValue(Coord, lat_or_lon = 'lat'):
    '''
        find coord - assumes NASA Nature run resolution of 0.0625
    '''
    if lat_or_lon.lower() == 'lat':
        Coord_Range = np.linspace(-90,90,2881)
    else:
        Coord_Range = np.linspace(-180,179.9375,5760)
    
    value = min(range(len(Coord_Range)), key=lambda i: abs(Coord_Range[i]-Coord))
    return value


def LatitudeChange(distance):
    '''
        Caculate the change in latitude with for a given distance travelled (in km)
        Assuming a perfectly spherical Earth
    '''
    earth_radius = 6378.1 # In km
    return np.rad2deg(distance/earth_radius)

def LongitudeChange(distance, latitude):
    '''
        Caculate the change in longitude with for a given distance travelled (in km)
        at a given latitude.
        Assuming a perfectly spherical Earth
    '''
    earth_radius = 6378.1 # In km
    r = earth_radius * np.cos(np.deg2rad(latitude))
    return np.rad2deg(distance/r)
    
### ============================================================================
### Ground station objects
### ============================================================================

class GroundStation:

    def __init__(self,lat,lon,altitude):
        '''
            Set the parameters for the ground station.
            Find the range of the model for 2 degree around the ground station.
        '''
        
        self.lat = lat
        self.lon = lon
        self.alt = altitude

        ## Get a two degree box around the ground station as this will be as far as the line of sight
        ## from a satellite will reach from the ground station.
        
        self.lat_min_index = GetCoordValue(lat - 2, lat_or_lon = 'lat')
        self.lat_max_index = GetCoordValue(lat + 2, lat_or_lon = 'lat')

        self.lon_min_index = GetCoordValue(lon - 2, lat_or_lon = 'lon')
        self.lon_max_index = GetCoordValue(lon + 2, lat_or_lon = 'lon')

        self.lat_index = GetCoordValue(lat, lat_or_lon = 'lat')
        self.lon_index = GetCoordValue(lon, lat_or_lon = 'lon')


def ImportData(G5NR_file_path,variable,timestamp):
    
    rounded_dt = RoundTime(timestamp)

    file_date = rounded_dt.strftime('%Y%m%d_%H%Mz')
    
    year = str(rounded_dt.year)
    month = str(rounded_dt.month).zfill(2)
    day = str(rounded_dt.day).zfill(2)

    full_file_name = '{}/inst30mn_3d_{}_Nv/Y{}/M{}/D{}/c1440_NR.inst30mn_3d_{}_Nv.{}.nc4'.format(
                    G5NR_file_path,variable, year,month,day,variable, file_date)
    
    dataset = Dataset(full_file_name, 'r')
    
    return dataset    

def GetModelLayerHeights(GrndStn,G5NR_file_path, timestamp):
    '''
        Returns the height of each layer above the previous based on pressure and temperature
        First layer is zero since its zero height above the surface. Needed for subsequent calculations
    '''

    pres = ImportData(G5NR_file_path, 'DELP',timestamp)
    delp = pres.variables['DELP'][0,:,GrndStn.lat_min_index:GrndStn.lat_max_index,GrndStn.lon_min_index:GrndStn.lon_max_index]
    
    temp = ImportData(G5NR_file_path,'T',timestamp)
    t = temp.variables['T'][0,:,GrndStn.lat_min_index:GrndStn.lat_max_index,GrndStn.lon_min_index:GrndStn.lon_max_index]

    layer_heights = np.zeros(t.shape)

    for x in range(t.shape[1]):
        for y in range(t.shape[2]):
            layer_heights[1:,x,y] = BoxHeights(delp[:,x,y],t[:,x,y])

    return layer_heights

def ModelLayerCoordList(grnd, azimuth, elevation_angle, layer_hgts):

    layer_num = [0]
    lat_layer_coord = [grnd.lat]
    lon_layer_coord = [grnd.lon]
    
    for layer in range(layer_hgts.shape[0])[1:]:
        # Find the height of the model cell then convert to km
        lat_index = grnd.lat_min_index - GetCoordValue(lat_layer_coord[-1],lat_or_lon = 'lat')
        lon_index = grnd.lon_min_index - GetCoordValue(lon_layer_coord[-1],lat_or_lon = 'lon')
        cell_height = layer_hgts[layer, lat_index, lon_index] / 1000
        
        horizontal_dist = cell_height / np.tan(np.deg2rad(elevation_angle))
        
        # Set a geographic point for the origin of the layer
        origin = geopy.Point(lat_layer_coord[-1], lon_layer_coord[-1])
        # Use geopy to calculate the destination
        destination = geodist(kilometers = horizontal_dist).destination(origin, azimuth)

        layer_num.append(layer)
        lat_layer_coord.append(destination.latitude)
        lon_layer_coord.append(destination.longitude)

    coord_df = pd.DataFrame({'LatEntry':lat_layer_coord,'LonEntry':lon_layer_coord}, index = layer_num)
    return coord_df

def FindIntersects(layer_df):

    layer_num = []
    lat_index = []
    lon_index = []

    for layer in layer_df.index[:-1]:
        lat_origin = GetCoordValue(layer_df.loc[layer].LatEntry, lat_or_lon = 'lat')
        lon_origin = GetCoordValue(layer_df.loc[layer].LonEntry, lat_or_lon = 'lon')

        lat_dest = GetCoordValue(layer_df.loc[layer+1].LatEntry, lat_or_lon = 'lat')
        lon_dest = GetCoordValue(layer_df.loc[layer+1].LonEntry, lat_or_lon = 'lon')

        intercepts_x, intercepts_y = line(lon_origin,lat_origin,lon_dest, lat_dest)
        
        for i in range(len(intercepts_x)):
            layer_num.append(layer)
            lat_index.append(intercepts_y[i])
            lon_index.append(intercepts_x[i])

    modelindex = pd.DataFrame({'Layer':layer_num, 'LatIndex':lat_index,'LonIndex':lon_index})

    return modelindex

def RoundTime(dt=None, roundTo= 30*60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 30 minutes.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)

if __name__ == '__main__':
    # Testers
    print('Running example')

    # Set the met file path. Currently has 4 placeholders (%s) for other information added later.
    # For using as an example I suggest just changing the directories to your local ones and either
    # using the same date as below or changing it to the date of the example file you have.
    # Example met file can be downloaded here:
    # https://g5nr.nccs.nasa.gov/data/DATA/0.0625_deg/inst/inst30mn_2d_met1_Nx/Y2006/M04/D03/c1440_NR.inst30mn_2d_met1_Nx.20060403_1200z.nc4
    Met_file_path = '/Users/dfinch/Documents/NASA_Nature/c1440_NR.inst30mn_%s_%s_%s.20060403_%sz.nc4'

    ## Calculate the height of each model layer above the given ground station
    hgt_layers = GetModelLayerHeights(grnd, Met_file_path)

    ## Returns lat and lon of the entry point for each layer
    layer_df = ModelLayerCoordList(grnd,az,el,hgt_layers)
    ## Converts lat and lon into grid coordinates and finds the cells intersected for each layer   
    index_df = FindIntersects(layer_df)

    print(index_df)
    
### ============================================================================
### END OF PROGRAM
### ============================================================================
