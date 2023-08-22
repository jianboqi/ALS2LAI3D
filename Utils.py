# coding: utf-8
from osgeo import gdal
import numpy as np
import mahotas
import random
import OpenEXR
import array, Imath
from xml.dom import minidom
import sys
import os


def get_less_install_dir():
    currdir = os.path.split(os.path.realpath(__file__))[0]
    if currdir.startswith(r"E:\03-Coding\lessrt"):
        return r"D:\LESS"
    else:
        return os.path.dirname(os.path.dirname(os.path.dirname(currdir)))

# writing arr to ENVI standard file format
def saveToHdr(npArray, dstFilePath, geoTransform=""):
    dshape = npArray.shape
    bandnum = 1
    format = "ENVI"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(dstFilePath, dshape[1], dshape[0], bandnum, gdal.GDT_Float32)
    if not geoTransform == "":
        dst_ds.SetGeoTransform(geoTransform)
    #     npArray = linear_stretch_3d(npArray)
    dst_ds.GetRasterBand(1).WriteArray(npArray)
    dst_ds = None


# only reading the data array
def read_img_to_arr_no_transform(img_file):
    data_set = gdal.Open(img_file)
    band = data_set.GetRasterBand(1)
    arr = band.ReadAsArray(0, 0, band.XSize, band.YSize)
    return arr, band.XSize, band.YSize


def quotes(param):
    return '"'+param+'"'


# reading image to area
def read_file_to_arr(img_file):
    dataset = gdal.Open(img_file)
    band = dataset.GetRasterBand(1)
    geoTransform = dataset.GetGeoTransform()
    dataarr = band.ReadAsArray(0, 0, band.XSize, band.YSize)
    return band.XSize, band.YSize, geoTransform[1], dataarr


def save_as_random_color_img(_dataarr, filepath):
    rows, cols = _dataarr.shape
    re = np.zeros((rows, cols, 3), dtype=np.uint8)
    colormap = dict()
    colormap[0.0] = [0, 0, 0]
    for row in range(rows):
        for col in range(cols):
            if _dataarr[row][col] in colormap:
                re[row, col, :] = colormap[_dataarr[row][col]]
            else:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                re[row, col, :] = color
                colormap[_dataarr[row][col]] = color
    mahotas.imsave(filepath, re)


def save_to_obj(vertices, faces, obj_path):
    f = open(obj_path, "w")
    for vertice in vertices:
        f.write("v %f %f %f\n" % (vertice[0], vertice[1], vertice[2]))
    for face in faces:
        f.write("f %d %d %d\n" % (face[0], face[1], face[2]))
    f.close()


def transmittance2lad_simple(T, d):
    # assuming the leaf angle distribution is spherical, and ingore the scanning angle
    if T < 0:
        return 0
    elif T == 0:
        return 1.0
    G = 0.5
    LAD = -np.log(T) / (G * d)
    return LAD


def transmittance2lai(T):
    if T < 0:
        return 0
    elif T == 0:
        return 5.0
    G = 0.5
    LAI = -np.log(T) / G
    return LAI


def write_voxel_obj(min_p, max_p, eps, obj_path):
    f = open(obj_path, "w")
    x_min, y_min, z_min = min_p
    x_max, y_max, z_max = max_p
    f.write("v %f %f %f\n" % (x_min + eps, y_min + eps, z_min + eps))
    f.write("v %f %f %f\n" % (x_min + eps, y_min + eps, z_max - eps))
    f.write("v %f %f %f\n" % (x_max - eps, y_min + eps, z_max - eps))
    f.write("v %f %f %f\n" % (x_max - eps, y_min + eps, z_min + eps))
    f.write("v %f %f %f\n" % (x_min + eps, y_max - eps, z_min + eps))
    f.write("v %f %f %f\n" % (x_min + eps, y_max - eps, z_max - eps))
    f.write("v %f %f %f\n" % (x_max - eps, y_max - eps, z_max - eps))
    f.write("v %f %f %f\n" % (x_max - eps, y_max - eps, z_min + eps))
    f.write("f %d %d %d %d\n" % (1, 4, 3, 2))
    f.write("f %d %d %d %d\n" % (5, 6, 7, 8))
    f.write("f %d %d %d %d\n" % (1, 5, 8, 4))
    f.write("f %d %d %d %d\n" % (1, 2, 6, 5))
    f.write("f %d %d %d %d\n" % (8, 7, 3, 4))
    f.write("f %d %d %d %d\n" % (7, 6, 2, 3))
    f.close()


def convert_dem_to_mip_monochromatic_chanel(dem_path, mip_path):
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    # save to openexr file
    XSize = band.XSize
    YSize = band.YSize
    dataarr = band.ReadAsArray(0, 0, band.XSize, band.YSize)
    dataarr = np.fliplr(dataarr)
    dataarr = np.reshape(dataarr, (XSize * YSize))
    dataarr = dataarr - dataarr.min()
    heightStr = array.array('f', dataarr).tobytes()

    data_dict = dict()
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    chaneldict = dict()
    data_dict["y"] = heightStr
    chaneldict["y"] = half_chan
    header = OpenEXR.Header(band.XSize, band.YSize)
    header["channels"] = chaneldict
    out = OpenEXR.OutputFile(mip_path, header)
    out.writePixels(data_dict)


def write_terrain_xml(xml_path, scene_width, scene_height):
    docNode = minidom.Document()
    rootNode = docNode.createElement("scene")
    rootNode.setAttribute("version", "0.5.0")
    docNode.appendChild(rootNode)
    emitter = docNode.createElement("emitter")
    emitter.setAttribute("type", "directional")
    rootNode.appendChild(emitter)
    shapeNode = docNode.createElement("shape")
    rootNode.appendChild(shapeNode)
    shapeNode.setAttribute("id", "terrain")
    shapeNode.setAttribute("type", "heightfield")
    strNode = docNode.createElement("string")
    shapeNode.appendChild(strNode)
    strNode.setAttribute("name", "filename")
    strNode.setAttribute("value", "dem.exr")
    tranNode = docNode.createElement("transform")
    shapeNode.appendChild(tranNode)
    tranNode.setAttribute("name", "toWorld")
    rotateNode = docNode.createElement("rotate")
    tranNode.appendChild(rotateNode)
    rotateNode.setAttribute("angle", "-90")
    rotateNode.setAttribute("x", "1")
    scaleNode = docNode.createElement("scale")
    tranNode.appendChild(scaleNode)
    scaleNode.setAttribute("x", str(0.5 * scene_width))
    scaleNode.setAttribute("z", str(0.5 * scene_height))
    xm = docNode.toprettyxml()
    with open(xml_path, "w") as f:
        f.write(xm)


def fix_normals(mesh):
    num_tris = mesh.triangles_center.shape[0]
    centroid = mesh.centroid
    for i in range(0, num_tris):
        tri_center = mesh.triangles_center[i]
        tri_normal = mesh.face_normals[i]
        dir = tri_center - centroid
        dir = dir / np.linalg.norm(dir)
        dot = np.dot(tri_normal, dir)
        if dot < 0:
            mesh.faces[i, [1, 2]] = mesh.faces[i, [2, 1]]


def log(*args):
    print(*args)
    sys.stdout.flush()
