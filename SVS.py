### ============================================================================
### Extract met data from the NASA Nature Model run files and put them in a
### managable format
### ============================================================================
from netCDF4 import Dataset
import math
import numpy as np
from datetime import datetime
import pandas as pd
from pyorbital import astronomy
import geopy
from geopy.distance import distance as geodist
from skimage.draw import line
### ============================================================================
'''
1) Calculate model layer thickness and model height
2) Calculate elevation and azimuth angle of the satellite from ground station
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

def GetObserverView(GrndStn, Satellite):
    '''
        This returns the azimuth angle and the elevation angle of a satellite for a given ground
        station. This calculation is copied from pyorbital module but with ammendments to
        work for this piece of code. This function still calls a function within pyorbital for
        some of the calculations.
    '''
    utc_time = np.datetime64(Satellite.time)
    
    (pos_x,pos_y,pos_z), (vel_x, vel_y, vel_z_) = astronomy.observer_position(
            utc_time, Satellite.lon, Satellite.lat, Satellite.alt)
                          
    (opos_x, opos_y, opos_z), (ovel_x, ovel_y, ovelz) = astronomy.observer_position(
            utc_time, GrndStn.lon, GrndStn.lat, GrndStn.alt)

    lon = np.deg2rad(GrndStn.lon)
    lat = np.deg2rad(GrndStn.lat)

    theta = (astronomy.gmst(utc_time) + lon) % (2 * np.pi)

    rx = pos_x - opos_x
    ry = pos_y - opos_y
    rz = pos_z - opos_z

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    top_s = sin_lat * cos_theta * rx + sin_lat * sin_theta * ry - cos_lat * rz
    top_e = -sin_theta * rx + cos_theta * ry
    top_z = cos_lat * cos_theta * rx + cos_lat * sin_theta * ry + sin_lat * rz

    az = np.arctan(-top_e/top_s)
    az = np.where(top_s > 0, az + np.pi, az)
    az = np.where(az < 0, az + 2 * np.pi, az)

    rg = np.sqrt(rx * rx + ry * ry + rz * rz)
    el = np.arcsin(top_z / rg)

    # Return azimuth angle and elevation angle
    return np.rad2deg(az), np.rad2deg(el)

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
### Ground station and satellite objects
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

class SatelliteOrbit:
    '''
        Set the satellite parameters
    '''

    def __init__(self, lat, lon, altitude, time):
        self.lat = lat
        self.lon = lon
        self.alt = altitude
        self.time = time


def ImportData(met_file,variable,timestamp = None):
    # Just using example file for now
    
    if not timestamp:
        if variable == 'met1':
            dim = '2d'
            suffix = 'Nx'
            time = '1200'
        else:
            suffix = 'Nv'
            dim = '3d'

        if variable == 'DELP':
            time = '1230'
        elif variable in ['PL','T']:
            time = '1300'

    dataset = Dataset(met_file % (dim, variable, suffix, time), 'r')
    
    return dataset    

def GetModelLayerHeights(GrndStn,met_file):
    '''
        Returns the height of each layer above the previous based on pressure and temperature
        First layer is zero since its zero height above the surface. Needed for subsequent calculations
    '''

    pres = ImportData(met_file, 'DELP')
    delp = pres.variables['DELP'][0,:,GrndStn.lat_min_index:GrndStn.lat_max_index,GrndStn.lon_min_index:GrndStn.lon_max_index]
    
    temp = ImportData(met_file,'T')
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

if __name__ == '__main__':
    # Testers
    print('Running example')

    
    Met_file_path = '/Users/dfinch/Documents/NASA_Nature/c1440_NR.inst30mn_%s_%s_%s.20060403_%sz.nc4'
    
    ## Lat, lon and altitude (in km) of ground station (currently set to Edinburgh)
    grnd = GroundStation(56,1,0.04)

    ## Lat, lon, alt & time of satellite (fairly random position chosen)
    ## For this example time does not make any difference so just set to current time
    satellite = SatelliteOrbit(54,3,400, datetime.now())

    ## Return the azimuth and elevation angle of the satellite from the point of the view of the ground station
    az,el = GetObserverView(grnd,satellite)

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
