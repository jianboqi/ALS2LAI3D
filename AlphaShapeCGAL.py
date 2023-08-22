# coding: utf-8
import ctypes
import numpy as np
import _ctypes
import trimesh
import os


class AlphaShapeCGAL(object):
    def __init__(self, tmp_dir, use_cgal):
        self.use_cgal = use_cgal
        if not use_cgal:
            return
        dir_path = os.path.dirname(os.path.realpath(__file__))
        curr_dir = os.getcwd()
        os.chdir(dir_path)
        self.ASA = ctypes.cdll.LoadLibrary(os.path.join(dir_path, 'ASA.dll'))
        os.chdir(curr_dir)
        self.tmp_dir = tmp_dir

    def alphashape(self, points, alpha):
        points = points.astype(np.float32)
        row = points.shape[0]
        if not points.flags['C_CONTIGUOUS']:
            points = points.ravel(order='C')
        do_ASA = self.ASA.do_ASA
        do_ASA.restype = ctypes.c_double
        tmp_obj = os.path.join(self.tmp_dir, "alpha_cgal_tmp.obj")
        volume = do_ASA(points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), row, bytes(tmp_obj, encoding='utf8'), ctypes.c_float(alpha))
        return trimesh.load(tmp_obj)

    def free_library(self):
        if self.use_cgal:
            _ctypes.FreeLibrary(self.ASA._handle)
