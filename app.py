from flask import Flask, render_template, request, jsonify
import numpy as np
from osgeo import gdal, gdalnumeric, ogr
from PIL import Image, ImageDraw
import shapefile
import simplejson
from sklearn.externals import joblib

def array_to_image(a):
    i = Image.fromstring('L',(a.shape[1], a.shape[0]),(a.astype('b')).tostring())
    return i

def image_to_array(i):
    a = gdalnumeric.fromstring(i.tostring(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a

def world_to_pixel(geo_matrix, x, y):
    ulX = geo_matrix[0]
    ulY = geo_matrix[3]
    xDist = geo_matrix[1]
    yDist = geo_matrix[5]
    rtnX = geo_matrix[2]
    rtnY = geo_matrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def get_shape_extent(points):
    maxX = max(points, key=lambda x: x[0])[0]
    minX = min(points, key=lambda x: x[0])[0]
    maxY = max(points, key=lambda x: x[1])[1]
    minY = min(points, key=lambda x: x[1])[1]
    return (minX, minY, maxX, maxY)

def get_ras_extent(ras): # xmin xmax, ymin, ymax
    tf = ras.GetGeoTransform()
    xL = tf[0]
    yT = tf[3]
    xR = xL + ras.RasterXSize*tf[1] # cols * width
    yB = yT + ras.RasterYSize*tf[5] # rows * height
    return (xL, yB, xR, yT) # xmin, ymin, xmax, ymax

def clip_raster(rast, gt, points):
    # from http://geospatialpython.com/
    minX, minY, maxX, maxY = get_shape_extent(points)
    # Convert the layer extent to image pixel coordinate
    ulX, ulY = world_to_pixel(gt, minX, maxY)
    lrX, lrY = world_to_pixel(gt, maxX, minY)
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    clip = rast[:, ulY:lrY, ulX:lrX]
    print [x for x in clip.shape]
    # Create a new geomatrix for the image
    gt2 = list(gt)
    gt2[0] = minX
    gt2[3] = maxY
    # Map points to pixels for drawing the boundary on a blank 8-bit,
    #   black and white, mask image.
    pixels = []
    for p in points:
        pixels.append(world_to_pixel(gt2, p[0], p[1]))
    raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(raster_poly)
    rasterize.polygon(pixels, 0) # Fill with zeroes
    mask = image_to_array(raster_poly)
    # Clip the image using the mask
    clip = gdalnumeric.choose(mask, (clip, 0)).astype(gdalnumeric.uint8)
    return clip

nn = joblib.load('static/modelFit/treemodel.pkl')

# Initialize the Flask application
app = Flask(__name__)

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the AJAX request and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/_getOutput', methods=['POST'])
def getOutput():
    # load shape
    shpe = request.json['xy']
    print shpe
    # x = simplejson.loads(shpe)
    # points = [[d[0],d[1]] for d in x['geometry']['coordinates'][0]]
    # # load raster
    # city = request.json['city']
    # #print city
    # if 'Washington' in city:
    #     fi='static/rast/DCmap.tif'
    # else:
    #     fi='static/rast/COmap.tif'
    # ras = gdal.Open(fi)
    # # verify in box
    # rex = get_ras_extent(ras)
    # pex = get_shape_extent(points)
    # if pex[0]<rex[0] or pex[1]<rex[1] or pex[2]>rex[2] or pex[3]>rex[3]:
    #     out = "Not in map bounds"
    # else:
    #     ra = ras.ReadAsArray()
    #     gt = ras.GetGeoTransform()
    #     ras2 = clip_raster(ra,gt,points)
    #     sr = np.log(ras2[3]*1./ras2[0])
    #     sr = np.ravel(sr[np.isfinite(sr)])
    #     preds = nn.predict(np.array(sr.reshape(-1,1)))
    #     out = [len(preds[preds=='T']),len(preds)] # pixels
    return jsonify(result='out') #result=out

if __name__ == '__main__':
    app.run(port=35507, debug=True)
