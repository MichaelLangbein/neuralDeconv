import json
import numpy as np
import shapely.geometry as shg
import sentinelsat as ss
import datetime as dt
import geopandas as gpd
import rasterio.features as riof
import rasterio.mask as riom
import rasterio.crs as rioc
import rasterio.transform as riot
from rasterio import MemoryFile
import utils as u


def readGeojsonToShapely(fileName):
    with open(fileName) as f:
        features = json.load(f)["features"]
    col = shg.GeometryCollection(
        [shg.shape(feature["geometry"]).buffer(0)
         for feature in features])
    return col


def downloadS5pData(startDate, endDate, footprint, targetDir='./downloads/', output='default'):
    """
        S5p is available under the same API as other S1-3, but under a different url and credentials.
        Instead of https://scihub.copernicus.eu/dhus/ use https://s5phub.copernicus.eu/dhus/, 
        and instead of your credentials use s5pguest/s5pguest.
    """

    api = ss.SentinelAPI('s5pguest', 's5pguest', 'https://s5phub.copernicus.eu/dhus/')
    products = api.query(
                    area=footprint,
                    area_relation='Intersects',
                    date=(startDate, endDate),
                    producttype = 'L2__NO2___',  #L2__CH4___
                    platformname='Sentinel-5',
                    processinglevel='L2'
    )
    api.download_all(products, targetDir)

    if output == 'default':
        return api, products
    if output == 'geopandas':
        return api, api.to_geodataframe(products)
    if output == 'pandas':
        return api, api.to_dataframe(products)
    if output == 'geojson':
        return api, api.to_geojson(products)



def getS5pData(startDate, endDate, footprint):
    """
        loads data from local nc files
    """


def getLkRaster(height, width):
    ger = readGeojsonToShapely('./data/germany_outline.geojson')
    west, south, east, north = ger.bounds
    affineTransform = riot.from_bounds(west, south, east, north, width, height)

    lks = gpd.read_file('./data/landkreise.geojson')
    data = [(geom, value) for geom, value in zip(lks.geometry, lks.index)]

    burned = riof.rasterize(
        data,
        out_shape=(height, width),
        transform=affineTransform,
        fill=0,
        dtype=np.float32
    )

    return (west, south, east, north), affineTransform, burned

    

def npArrayToRasterioDataset(npArray, crs, affineTransform):
    height, width = npArray.shape
    npArray = npArray.reshape((1, height, width))

    profile = {
        'driver': 'GTiff', 
        'dtype': npArray.dtype, 
        'width': width, 
        'height': height,
        'count': 1,
        'crs': rioc.CRS.from_epsg(crs), 
        'transform': affineTransform, 
        'tiled': False,
        'nodata': 0
       }
    
    memfile = MemoryFile()
    dataset = memfile.open(**profile)
    dataset.write(npArray)
    dataset.close()
    return memfile.open()



def getLksRasterioDataset(width, height):
    footprint, transform, data = getLkRaster(height, width)
    return npArrayToRasterioDataset(data, 4326, transform)



def getWindowFromRasterioDataset(west, south, east, north, dataset):
    box = shg.box(west, south, east, north)
    cropped, _ = riom.mask(dataset, [box], crop=True)
    return cropped

    