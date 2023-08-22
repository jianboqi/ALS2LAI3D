# coding: utf-8
# author: Jianbo Qi from Beijing Normal University
# Date: 2022/5/12  (Deeply mourn the Victims in the Sichuan Earthquake 2008.5.12)
# This script convert a point cloud (las file) into 3D scenes, ground filtering, crown segmentation,
# leaf area density inversion etc. are all included, and automatically executed.
# Contact: jianboqi@bnu.edu.cn

import argparse
from AlsScene import AlsScene
import Utils

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # required parameters
    parse.add_argument("-i", help="Input file.", required=True)
    parse.add_argument("-o", help="Output Directory for storing LESS simulation project.", required=True)
    parse.add_argument("-sim_name", help="LESS simulation name", default="less_sim_proj")
    # optional parameters
    parse.add_argument('-type', help="LAD estimation type: voxel, alphashape", default="alphashape")
    parse.add_argument('-seg_method', help="The method to segment trees: watershed, kmeans, hexagon, kmeans_points", default="watershed")
    parse.add_argument('-seg_param', help="The parameter for segmentation", type=str, default="default")
    parse.add_argument("-param", help="Voxel resolution or alphashape alpha value", type=str, default="10")
    parse.add_argument("-include_understory", help="If understory is included", type=str, default="true")
    parse.add_argument('-pad_method', help="The method to estimate plant area density: pulse_tracing, point_number, "
                                           "user_defined, total_lai", default="pulse")
    parse.add_argument('-total_lai', help="Constrain the total LAI", type=str, default="3.0")
    parse.add_argument("-pad_constant", help="PAD constant value when pad_method equals to user_defined", type=float, default=0.8)
    parse.add_argument("-pad_scale", help="Scale factor of plant area density", type=float, default=1.0)
    parse.add_argument("-understory_height", help="understory height devision", type=str, default="auto")
    parse.add_argument('-leaf_angle_dist', help="Leaf Angle Distribution", type=str, default="Spherical")
    parse.add_argument('-leaf_op', help="Leaf optical properties", type=str, default="birch_leaf_green")
    parse.add_argument("-leaf_as_facet", help="If leaf is represented as facet", type=str, default="false")
    parse.add_argument("-single_leaf_area", help="Single leaf area", type=float, default=0.01)
    args = parse.parse_args()

    input_las = args.i
    out_dir = args.o
    less_sim_proj = args.sim_name
    estimation_type = args.type
    param = args.param
    leaf_angle_dist = args.leaf_angle_dist
    leaf_op = args.leaf_op
    seg_method = args.seg_method
    if args.include_understory == "true":
        include_understory = True
    else:
        include_understory = False
    pad_scale = args.pad_scale
    pad_method = args.pad_method
    pad_constant = args.pad_constant
    understory_height = args.understory_height
    seg_param = args.seg_param
    total_lai_mode = 0  # 0: equal pad, 1: pulse tracing and constrain total lai, 2: point number and constrain lai
    total_lai = 0
    if pad_method == "total_lai":
        if "/" in args.total_lai:
            arr = args.total_lai.split("/")
            total_lai = float(arr[0])
            total_lai_mode = int(arr[1])
            if total_lai_mode == 1:
                pad_method = "pulse_tracing"
            elif total_lai_mode == 2:
                pad_method = "point_number"
            elif total_lai_mode == 0:
                pad_method = "total_lai"
            else:
                pad_method = "total_lai"
        else:
            total_lai = float(args.total_lai)

    if args.leaf_as_facet == "true":
        leaf_as_facet = True
    else:
        leaf_as_facet = False
    single_leaf_area = args.single_leaf_area

    als_scene = AlsScene(input_las, out_dir, less_sim_proj, estimation_type, param,
                         leaf_angle_dist, leaf_op, seg_method, include_understory, pad_scale, pad_method, pad_constant,
                         understory_height, seg_param, total_lai, total_lai_mode, leaf_as_facet, single_leaf_area)
    als_scene.set_less_install_dir(Utils.get_less_install_dir())
    als_scene.convert()

