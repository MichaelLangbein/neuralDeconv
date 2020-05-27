import json
import numpy as np
import shapely.geometry as shg
import sentinelsat as ss
import datetime as dt
import geopandas as gpd
import rasterio as rio
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
    dataFeb = rio.open('netcdf:./downloads/data_2020_feb.nc:tcno2', 'r')
    dataMar = rio.open('netcdf:./downloads/data_2020_mar.nc:tcno2', 'r')
    return dataFeb, dataMar


def getWindowFromS5pRasterioDataset(startDate, endDate, west, south, east, north, data):
    bbox = shg.box(west, south, east, north)
    delta = dt.timedelta(0, 6*60*60)
    start = dt.datetime(startDate.year, startDate.month, 1)
    dateIndexMap = {start + i*delta: i+1 for i in range(len(data.indexes))}
    i0 = dateIndexMap[startDate]
    i1 = dateIndexMap[endDate]
    window, transf = riom.mask(data, [bbox], crop=True, indexes=[i for i in range(i0, i1+1)])
    return window



def getBoundsGermany():
    ger = readGeojsonToShapely('./data/germany_outline.geojson')
    west, south, east, north = ger.bounds
    return shg.box(west, south, east, north)


def getLkRaster(height, width):
    bbox = getBoundsGermany()
    west, south, east, north = bbox.bounds
    affineTransform = riot.from_bounds(west, south, east, north, width, height)

    lks = gpd.read_file('./data/landkreise_fallzahlen.json')
    lks['density'] = lks['population'] / lks.geometry.area / 1000000
    data = [(geom, value) for geom, value in zip(lks.geometry, lks['density'])]

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



def rioGetShapes(dataset):
    w = dataset.meta['width']
    h = dataset.meta['height']
    c = dataset.meta['count']
    bounds = dataset.bounds
    west = bounds.left
    south = bounds.bottom
    east = bounds.right
    north = bounds.top
    return (w, h, c), shg.box(west, south, east, north)


def intersectGTandS5(groundTruthDataset, s5pDataset):
    (cs, rs, ts), bboxS = rioGetShapes(s5pDataset)
    (cg, rg, tg), bboxG = rioGetShapes(groundTruthDataset)
    intersection = bboxS.intersection(bboxG)
    groundTruthIS = riom.mask(groundTruthDataset, [intersection], crop=True)
    s5pDataIS = riom.mask(s5pDataset, [intersection], crop=True)
    return groundTruthIS, s5pDataIS


def createTrainingPair(windowSize, timeSpan, npGroundTruthData, npS5Data):
    T, R, C = npS5Data.shape
    t = np.random.randint(T - timeSpan)
    r = np.random.randint(R - windowSize)
    c = np.random.randint(C - windowSize)
    gtWindow = npGroundTruthData[r:r+windowSize, c:c+windowSize]
    s5Window = npS5Data[t:t+timeSpan, r:r+windowSize, c:c+windowSize]
    return s5Window, gtWindow


def getShapelyBounds(dataset):
    bounds = dataset.bounds
    bbx = shg.box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return bbx


def trainingDataGenerator(batchSize, imageSize, timeSpan):
    bboxGermany = getBoundsGermany()
    s5dataSet = rio.open('netcdf:./downloads/data_2020_feb.nc:tcno2', 'r')
    s5data, transf = riom.mask(s5dataSet, [bboxGermany], crop=True)
    T, R, C = s5data.shape
    print(f"Study area: {T} * {R} * {C}")
    _, _, gtData = getLkRaster(R, C)
    while True:
        xs = np.zeros((batchSize, imageSize, imageSize, 1))
        ys = np.zeros((batchSize, imageSize, imageSize, 1))
        for i in range(batchSize):
            s5Window, gtWindow = createTrainingPair(imageSize, timeSpan, gtData, s5data)
            s5WindowMean = np.mean(s5Window, axis=0)
            gtMax = np.max(gtWindow)
            s5Max = np.max(s5WindowMean)
            gtWindow /= gtMax
            s5WindowMean /= s5Max
            xs[i, :, :, 0] = s5WindowMean # observation
            ys[i, :, :, 0] = gtWindow # truth
        yield xs, ys