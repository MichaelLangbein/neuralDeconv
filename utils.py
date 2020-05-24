import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
import scipy.ndimage as scn
import scipy.signal as scs
import scipy.fftpack as scf
import scipy.stats as scst
import skimage.restoration as sir
import netCDF4 as nc
import keras as k
import json
import shapely as sh
import sentinelsat as ss
import datetime as dt


def readGeojsonToShapely(fileName):
    with open(fileName) as f:
        features = json.load(f)["features"]
    col = sh.geometry.GeometryCollection(
        [sh.geometry.shape(feature["geometry"]).buffer(0)
         for feature in features])
    return col


def downloadS5pData(startDate, endDate, 
                    targetDir='./downloads/', uname='s5pguest', pw='s5pguest', 
                    url='https://scihub.copernicus.eu/dhus'):

    ger = readGeojsonToShapely('./data/germany_outline.geojson')
    footprint = sh.geometry.box(*(ger.bounds))

    api = ss.SentinelAPI(uname, pw, url)
    products = api.query(footprint.wkt,
                     date=(startDate, endDate),
                     platformname='Sentinel-5 Precursor',
                     cloudcoverpercentage=(0, 30))
    api.download_all(products, targetDir)
    
    # convert to Pandas DataFrame
    # products_df = api.to_dataframe(products)

    return products




def tabling(f):
    tabling.table = []
    def wrapped(*args, **kwargs):
        for (targs, tkwargs), tresult in tabling.table:
            if args == targs and kwargs == tkwargs:
                return tresult
        result = f(*args, **kwargs)
        tabling.table.append(((args, kwargs), result))
        return result
    wrapped.__name__ = 'w' + f.__name__
    return wrapped


def logging(f):
    logging.depth = 0
    def wrapped(*args, **kwargs):
        print(f"{'|---' * logging.depth}{f.__name__}{args}")
        logging.depth += 1
        result = f(*args, **kwargs)
        logging.depth -= 1
        print(f"{'|---' * logging.depth}{f.__name__}{args} => {result}")
        return result
    return wrapped



def round(val):
    return int(val + 0.5)


def distance(v1, v2):
    return np.hypot(v1, v2)


def unitdoughnut(r):
    d = 2*r + 1
    rx, ry = d/2, d/2
    x, y = np.indices((d, d))
    return (np.abs(np.hypot(rx - x, ry - y)-r) < 0.5).astype(int)


def point(R, C, cx, cy, radius):
    xx, yy = np.mgrid[:R, :C]
    distance = np.hypot(xx - cx, yy - cy) 
    circle = distance <= radius
    return circle


def rotate(image, degrees):

    # rotated
    rotated = scn.rotate(image, degrees)

    # crop
    r, c = image.shape
    R, C = rotated.shape
    deltaR = int((R - r) / 2)
    deltaC = int((C - c) / 2)
    if deltaR > 0 or deltaC > 0:
        rotated = rotated[deltaR:-deltaR, deltaC:-deltaC]

    return rotated


def line(R, C, sr, sc, er, ec):
    dR = er - sr
    dC = ec - sc
    if dC == 0 and dR != 0:
        m = line(C, R, sc, sr, ec, er)
        m = rotate(m, 90)
        return m
    else:
        if dC == 0 and dR == 0:
            dRdC = 0
        else:
            dRdC = dR / dC
        m = np.zeros((R, C))
        for c in range(sc, ec +1):
            r = sr + dRdC * (c - sc)
            r = round(r)
            m[r, c] = 1
        return m


def centerPad(kernel, n):
    l, _ = kernel.shape
    m = np.zeros((n, n))
    delta = int((n - l) / 2)
    m[delta:delta+l, delta:delta+l] = kernel
    return m


def rotationMatrix(theta):
    return np.array([
        [np.cos(theta), - np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def conv(kernel, matrix):
    return scs.convolve2d(matrix, kernel, 'same')


def motionBlurKernel(size, angle):
    l = line(size, size, round(size/2), 0, round(size/2), size-1)
    k = rotate(l, angle)
    return k


def gaussianKernel(size, s1, s2, cov):
    Sigma = [
        [s1, cov],
        [cov, s2]
    ]
    rv = scst.multivariate_normal([0, 0], Sigma)
    m = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            x = c - round(size/2)
            y = r - round(size/2)
            m[r, c] = rv.pdf([x, y])
    return m


def whiteNoise(R, C, span):
    n = (np.random.rand(R, C) - 0.5) * span
    return n


def neighborhood(image, r, c, d):
    return image[r-d:r+d, c-d:c+d]


def grow(image, radius):
    m = np.zeros(image.shape)
    for r, row in enumerate(image):
        for c, el in enumerate(row):
            nbh = neighborhood(image, r, c, radius)
            sum = np.sum(nbh)
            if sum > 0:
                m[r, c] = 1
    return m


def randomImage(size, nrNodes):
    m = np.zeros((size, size))
    centers = [(np.random.randint(size), np.random.randint(size)) for i in range(nrNodes)]
    for center in centers:
        m += point(size, size, center[0], center[1], np.random.randint(10))
    for i in range(nrNodes-1):
        c1 = centers[i]
        c2 = centers[i+1]
        m += line(size, size, c1[0], c1[1], c2[0], c2[1])
    return m



def createTrainingPair(size, maxSpread=10, maxNodes=10):
    nrNodes = np.random.randint(maxNodes) + 1
    s1 = np.random.randint(maxSpread) + 1
    s2 = np.random.randint(maxSpread) + 1
    cv = np.random.rand()
    angle = np.random.randint(180)

    base = randomImage(size, nrNodes)
    k = gaussianKernel(maxSpread*2, s1, s2, cv)
    k = rotate(k, angle)
    im = conv(k, base)

    return base, im



def createSimpleAutoencoder(imgWidth, imgHeight, nodes=[700, 500, 300, 100]):
    obsSize = imgWidth * imgHeight
    sparseSize = nodes[-1]

    input = k.Input([obsSize])
    x = input
    for n in nodes:
        x = k.layers.Dense(n, activation='sigmoid', name=f"encoder_d{n}")(x)
    encoder = k.Model(input, x, name='encoder')

    decInput = k.Input([sparseSize])
    y = decInput
    for n in nodes[::-1]:
        y = k.layers.Dense(n, activation='sigmoid', name=f"decoder_d{n}")(y)
    y = k.layers.Dense(obsSize, activation='sigmoid', name=f"decoder_d{n+1}")(y)
    decoder = k.Model(decInput, y, name='decoder')

    autoencoder = k.Model(input, decoder(encoder(input)), name='autoencoder')

    return encoder, decoder, autoencoder



def createConv2Autoencoder(imgHeight, imgWidth, channels, nodes=[]):
    
    input = k.Input(shape=(imgHeight, imgWidth, channels))

    x = input
    for i, n in enumerate(nodes):
        x = k.layers.Conv2D(n, (3, 3), activation='relu', padding='same', name=f"encoder_conv{i}")(x)
        x = k.layers.MaxPooling2D((2, 2), padding='same', name=f"encoder_maxp{i}")(x)

    y = x
    for i, n in enumerate(nodes[::-1]):
        y = k.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name=f"decoder_conv{i}")(y)
        y = k.layers.UpSampling2D((2, 2), name=f"decoder_upspl{i}")(y)
    y = k.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name=f"decoder_conv{i+1}")(y)

    autoencoder = k.Model(input, y)

    return autoencoder