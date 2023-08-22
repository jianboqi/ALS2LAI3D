# coding: utf-8
# coding: utf-8
# author: Jianbo Qi from Beijing Normal University
# Date: 2022/5/12  (Deeply mourn the Victims in the Sichuan Earthquake 2008.5.12)
# This script convert a point cloud (las file) into 3D scenes, ground filtering, crown segmentation,
# leaf area density inversion etc. are all included, and automatically executed.
# Contact: jianboqi@bnu.edu.cn

import os
import CSF
import sys
from sklearn.cluster import KMeans
import Utils
import mahotas
import alphashape
import trimesh
from xml.dom import minidom
from Pulse import *
from sklearn.neighbors import NearestNeighbors
import glob
from csfnormalize import csf_normalize_internal
from hexagonify import hexagonify
import logging
from LeafAsFacet import LeafAsFacetGenerator
from AlphaShapeCGAL import AlphaShapeCGAL
import subprocess


class AlsScene(object):
    def __init__(self, las_path, out_dir, less_sim_proj, estimation_type, param, leaf_angle_dist, leaf_op,
                 seg_method, include_understory, pad_scale, pad_method, pad_constant, understory_height, seg_param,
                 total_lai, total_lai_mode, leaf_as_facet, single_leaf_area):
        self.las_path = las_path
        self.root_dir = os.path.dirname(las_path)
        self.tmp_path = os.path.join(self.root_dir, "_tmp_")
        self.less_sim_path = os.path.join(out_dir, less_sim_proj)
        self.less_root_dir = r"D:\LESS"
        self.estimation_type = estimation_type
        self.pad_est_type = "average_pad"
        arr = param.split("/")
        self.param = float(arr[0])
        self.use_cgal = False if len(arr) > 1 else True
        self.leaf_angle_dist = leaf_angle_dist
        self.leaf_op = leaf_op
        self.seg_method = seg_method
        self.seg_param = seg_param
        self.include_understory = include_understory
        self.pad_scale = pad_scale
        self.pad_method = pad_method
        self.pad_constant = pad_constant
        self.understory_height = understory_height
        self.total_lai = total_lai
        self.total_lai_mode = total_lai_mode
        self.leaf_as_facet = leaf_as_facet
        self.single_leaf_area = single_leaf_area
        self.__init_dir()

        self.ptCloud = None
        self.scene_width = 100
        self.scene_height = 100
        self.scene_thick = 30

        self.all_pulses = []
        self.scene = None
        self.terr_scene = None
        self.RAY_O_Z = 1000
        self.pad = dict()

    def convert(self):
        self.__read_point_cloud()
        self.__do_ground_filtering()
        self.__generate_dem_dsm()
        if self.estimation_type == "alphashape" or self.estimation_type == "ellipsoid" \
                or self.estimation_type == "cone" or self.estimation_type == "cylinder":
            self.__seg_point_cloud()
        elif self.estimation_type == "voxel":
            self.__voxlization_point_cloud()
        else:
            Utils.log("Invalid parameter 'type'")
            sys.exit(0)
        is_pad_succeed = False
        if self.pad_method == "pulse_tracing":
            if self.pad_est_type == "average_pad":
                is_pad_succeed = self.__pad_inversion_average_pad()
            else:
                is_pad_succeed = self.__pad_inversion_average_path()
        elif self.pad_method == "point_number":
            is_pad_succeed = self.__pad_point_number()
        elif self.pad_method == "user_defined":
            is_pad_succeed = self.__pad_user_defined()
        elif self.pad_method == "total_lai":
            is_pad_succeed = self.__pad_by_total_lai()

        if is_pad_succeed:
            self.__create_less_simulation()
            Utils.log(" - Finished.")

    def set_less_install_dir(self, less_install_dir):
        self.less_root_dir = less_install_dir

    def __init_dir(self):
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)

        # if os.path.exists(self.less_sim_path):
        #     shutil.rmtree(self.less_sim_path, ignore_errors=True)

    def __read_point_cloud(self):
        self.ptCloud = laspy.read(self.las_path)
        Utils.log(" - Reading LAS file...")
        Utils.log(" - LAS version:", self.ptCloud.header.version, "Point Format:", self.ptCloud.point_format.id)
        if self.pad_method == "pulse_tracing" and (self.ptCloud.point_format.id == 0 or self.ptCloud.point_format.id == 2):
            Utils.log(" - Point format of LAS is %d, it does not have gps time, and is not supported currently."
                      % self.ptCloud.point_format.id)
            sys.exit(0)
        bounds = self.ptCloud.header.max - self.ptCloud.header.min
        self.scene_width = bounds[0]
        self.scene_height = bounds[1]
        self.scene_thick = bounds[2]

    def __do_ground_filtering(self):
        Utils.log(" - Perform ground filtering...")
        csf = CSF.CSF()  # 构造CSF
        csf.params.cloth_resolution = 0.4
        csf.params.bSlopeSmooth = True
        csf.params.rigidness = 1
        csf.params.class_threshold = 0.3
        csf.params.rasterization_mode = 1
        csf.params.rasterization_window_size = 10
        csf.downsampling_window_num = 5

        csf.setPointCloud(self.ptCloud.xyz)
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground, False)
        classification = np.array([1 for i in range(len(self.ptCloud.points))])
        classification[ground] = 2
        classification[non_ground] = 1  # 1 for non ground point
        self.ptCloud.classification = classification

    def __generate_dem_dsm(self):
        Utils.log(" - Generate DEM/DSM...")
        dem_path = os.path.join(self.tmp_path, "dem")
        dsm_path = os.path.join(self.tmp_path, "dsm")
        chm_path = os.path.join(self.tmp_path, "chm")
        # ground_points_xyz = self.ptCloud.xyz[self.ptCloud.classification == 2]
        # o3d_ptCloud = o3d.geometry.PointCloud()
        # o3d_ptCloud.points = o3d.utility.Vector3dVector(ground_points_xyz)
        # o3d.visualization.draw_geometries([o3d_ptCloud])
        out_classified_las_path = os.path.join(self.tmp_path, "classified.las")
        ground_las = laspy.LasData(self.ptCloud.header)
        ground_las.points = self.ptCloud.points
        ground_las.write(out_classified_las_path)
        curr_dir = os.path.split(os.path.realpath(__file__))[0]
        # cmd = [Utils.quotes(sys.executable), os.path.join(curr_dir, "csfdem.py"), "-i", out_classified_las_path, "-o", dem_path,
        #        "-dsm", dsm_path, "-chm", chm_path, "-resolution", "0.5"]
        # # Utils.log("  - "+" ".join(cmd))
        # os.system(" ".join(cmd))
        subprocess.call(
            [sys.executable, os.path.join(curr_dir, "csfdem.py"), "-i", out_classified_las_path, "-o", dem_path,
             "-dsm", dsm_path, "-chm", chm_path, "-resolution", "0.5"])

    def __get_normalize_point_under_overstory_division_height(self):
        Utils.log(" - Normalizing point cloud...")
        dem_path = os.path.join(self.tmp_path, "dem")
        intput_las = os.path.join(self.tmp_path, "classified.las")
        out_norm_las = os.path.join(self.tmp_path, "normalized.las")
        normalized_pt = csf_normalize_internal(intput_las, dem_path)
        normalized_pt.write(out_norm_las)  # write the normalized point cloud for debug
        idx = (normalized_pt.z/0.5).astype(int)
        unique, counts = np.unique(idx, return_counts=True)
        one_third_height = int(len(unique)*1/3)
        d = dict(zip(unique[0: one_third_height], counts[0: one_third_height]))
        min_idx = min(d, key=d.get)
        division_height = (min_idx+1)*0.5  # below this height is understory
        veg_points_z = normalized_pt.z[normalized_pt.classification == 1]
        return division_height, veg_points_z < division_height

    def __voxlization_point_cloud(self):
        Utils.log(" - Voxelize point cloud...")
        num_width = int(self.scene_width/self.param+0.5)
        num_height = int(self.scene_height/self.param+0.5)
        num_layer = int(self.scene_thick/self.param+0.5)
        Utils.log("  - Voxel dimensions:", num_width, num_height, num_layer)
        veg_points = self.ptCloud.xyz[self.ptCloud.classification == 1]
        veg_points = veg_points - self.ptCloud.header.min
        veg_indices = np.floor(veg_points/self.param)
        ground_points = self.ptCloud.xyz[self.ptCloud.classification == 2]
        ground_points = ground_points - self.ptCloud.header.min
        ground_indices = np.floor(ground_points/self.param)
        index = 1
        docNode = minidom.Document()
        rootNode = docNode.createElement("scene")
        rootNode.setAttribute("version", "0.5.0")
        docNode.appendChild(rootNode)
        emitter = docNode.createElement("emitter")
        emitter.setAttribute("type", "directional")
        rootNode.appendChild(emitter)

        obj_files = glob.glob(os.path.join(self.tmp_path, "*.obj"))
        for obj_file in obj_files:
            os.remove(obj_file)

        fvol = open(os.path.join(self.tmp_path, "shapes.txt"), "w")
        for w in range(num_width):
            if w > 0 and w%5 == 0:
                Utils.log("   - percentage: %.1f%s" % (100 * (w/(num_width-1)), "%"))
            if w == num_width-1:
                Utils.log("   - percentage: %.1f%s" % (100, "%"))
            for h in range(num_height):
                indices = (veg_indices[:, 0] == w) & (veg_indices[:, 1] == h)
                layer_indice = veg_indices[indices, 2]
                layers = set(layer_indice)
                num_ground_point_in_cell = len(ground_indices[(ground_indices[:, 0] == w) & (ground_indices[:, 1] == h)])
                for layer in layers:
                    num_veg_bottom = len(layer_indice[layer_indice < layer])
                    num_veg_current = len(layer_indice[layer_indice == layer])
                    T = (num_ground_point_in_cell+num_veg_bottom)/(num_ground_point_in_cell+num_veg_current+num_veg_bottom)
                    lad = Utils.transmittance2lad_simple(T, self.param)
                    x_max = 0.5*self.scene_width - w*self.param
                    x_min = 0.5 * self.scene_width - (w + 1) * self.param
                    y_min = layer * self.param
                    y_max = (layer + 1) * self.param
                    z_min = h * self.param - 0.5*self.scene_height
                    z_max = (h+1) * self.param - 0.5*self.scene_height
                    fvol.write(str(index) + " " + str(self.param*self.param*self.param) + " " + str(lad) + "\n")
                    obj_path = os.path.join(self.tmp_path, str(index)+".obj")
                    Utils.write_voxel_obj([x_min, y_min, z_min], [x_max, y_max, z_max], 0.0001, obj_path)
                    shape = docNode.createElement("shape")
                    shape.setAttribute("type", "obj")
                    shape.setAttribute("id", str(index))
                    rootNode.appendChild(shape)
                    strdoc = docNode.createElement("string")
                    strdoc.setAttribute("name", "filename")
                    strdoc.setAttribute("value", str(index) + ".obj")
                    shape.appendChild(strdoc)
                    index += 1
        xm = docNode.toprettyxml()
        with open(os.path.join(self.tmp_path, "shapes.xml"), "w") as f:
            f.write(xm)
        fvol.close()

    def __seg_point_cloud(self):
        Utils.log(" - Segment point cloud...")
        Utils.log("  - Segmentation Method:", self.seg_method)
        Utils.log("  - Estimation Type:", self.estimation_type)
        chm_path = os.path.join(self.tmp_path, "chm")
        division_height, understory_idx = self.__get_normalize_point_under_overstory_division_height()
        Utils.log("  - Understory and overstory division height:", division_height)
        if self.understory_height == "auto":
            height_threshold = division_height
        else:
            height_threshold = float(self.understory_height)
        x_size, y_size, resolution,chm_data = Utils.read_file_to_arr(chm_path)
        window_size = 9
        if self.seg_method == "watershed":
            window_size = int(float(self.seg_param) / resolution)
        threshed = (chm_data > height_threshold)
        chm_data *= threshed
        bc = np.ones((window_size, window_size))

        maxima = mahotas.morph.regmax(chm_data, Bc=bc)
        spots, n_spots = mahotas.label(maxima)

        surface = (chm_data.max() - chm_data)
        areas = mahotas.cwatershed(surface, spots)
        areas *= threshed
        Utils.save_as_random_color_img(areas, os.path.join(self.tmp_path, "seg.jpg"))
        area_max = areas.max()
        Utils.log("  - Number of detected trees: ", area_max)

        veg_points = self.ptCloud.xyz[self.ptCloud.classification == 1]
        veg_points = veg_points - self.ptCloud.header.min
        # added by jianbo qi, 2023.2.5
        ground_points = self.ptCloud.xyz[self.ptCloud.classification == 2]  # ground points
        ground_points = ground_points - self.ptCloud.header.min

        overstory_labels = None
        ground_point_labels = None
        understory_labels = None
        if self.seg_method == "watershed":
            veg_indices = np.floor(veg_points[:, 0:2] / resolution).astype(int)
            veg_indices[:, 1] = y_size - veg_indices[:, 1] - 1
            point_labels = areas[veg_indices[:, 1], veg_indices[:, 0]]
            overstory_labels = point_labels[~understory_idx]
            understory_labels = point_labels[understory_idx]

            ground_indices = np.floor(ground_points[:, 0:2] / resolution).astype(int)
            ground_indices[:, 1] = y_size - ground_indices[:, 1] - 1
            ground_point_labels = areas[ground_indices[:, 1], ground_indices[:, 0]]

        # change to less internal coordinate system
        veg_points[:, [2, 1]] = veg_points[:, [1, 2]]
        veg_points[:, 0] = 0.5 * self.scene_width - veg_points[:, 0]
        veg_points[:, 2] = veg_points[:, 2] - 0.5 * self.scene_height

        ground_points[:, [2, 1]] = ground_points[:, [1, 2]]
        ground_points[:, 0] = 0.5 * self.scene_width - ground_points[:, 0]
        ground_points[:, 2] = ground_points[:, 2] - 0.5 * self.scene_height
        # np.savetxt(os.path.join(self.tmp_path, "vegcloud.txt"), veg_points)

        overstory_points = veg_points[~understory_idx]
        understory_points = veg_points[understory_idx]
        Utils.log("  - Understory and overstory points:", len(understory_points), len(overstory_points))

        alpha = 1/self.param
        index = 1
        docNode = minidom.Document()
        rootNode = docNode.createElement("scene")
        rootNode.setAttribute("version", "0.5.0")
        docNode.appendChild(rootNode)
        emitter = docNode.createElement("emitter")
        emitter.setAttribute("type", "directional")
        rootNode.appendChild(emitter)

        obj_files = glob.glob(os.path.join(self.tmp_path, "*.obj"))
        for obj_file in obj_files:
            os.remove(obj_file)

        fvol = open(os.path.join(self.tmp_path, "shapes.txt"), "w")
        logging.getLogger().setLevel(logging.ERROR)
        if self.include_understory and len(understory_points) > 100:
            alpha_cg = AlphaShapeCGAL(self.tmp_path, self.use_cgal)
            under_res = 5  # resolution to segment understory
            Utils.log("  - Segmenting understory, please wait...")
            under_indices = np.floor(understory_points[:, [0, 2]]/under_res)
            ground_indices = np.floor(ground_points[:, [0, 2]]/under_res)
            max_indices = under_indices.max(axis=0).astype(int)
            min_indices = under_indices.min(axis=0).astype(int)
            tot_y = (max_indices[1] - min_indices[1] + 1)
            tot = (max_indices[0]-min_indices[0] + 1) * tot_y
            sys.stdout.flush()
            meshes = []
            pad = []
            index_i = 0
            interval = 1 if tot <= 10 else int(tot / 10)
            for w in range(min_indices[0], max_indices[0]+1):
                # if index_i % (2*tot_y) == 0:
                #     percentage = index_i / tot
                #     Utils.log("   - percentage: %.0f%s" % (100 * percentage, "%"))
                for h in range(min_indices[1], max_indices[1]+1):
                    # if index_i == tot - 1:
                    #     Utils.log("   - percentage: 100%")
                    if index_i % interval == 0:
                        percentage = index_i / tot
                        Utils.log("   - percentage: %.0f%s" % (100 * percentage, "%"))
                    if index_i == tot - 1:
                        Utils.log("   - percentage: 100%")

                    index_i += 1
                    indices = (under_indices[:, 0] == w) & (under_indices[:, 1] == h)
                    num_ground_in_cell = len(ground_indices[(ground_indices[:, 0] == w) & (ground_indices[:, 1] == h)])
                    subpoints = understory_points[indices]
                    if len(subpoints) > 10:
                        try:
                            if self.use_cgal:
                                mesh = alpha_cg.alphashape(subpoints, self.param)
                            else:
                                mesh = alphashape.alphashape(subpoints, alpha)

                            if len(mesh.faces) >= 3:
                                meshes.append(mesh)
                                # compute leaf area density
                                T = num_ground_in_cell / (num_ground_in_cell + len(subpoints))
                                pad.append(Utils.transmittance2lai(T)*under_res*under_res/mesh.volume)
                                if not mesh.is_watertight:
                                    Utils.fix_normals(mesh)
                        except Exception as e:
                            pass
            alpha_cg.free_library()
            for mesh in meshes:
                if len(mesh.faces) >= 3:
                    obj = trimesh.exchange.obj.export_obj(mesh)
                    fvol.write(str(index) + " " + str(mesh.volume) + " " + str(pad[index-1]) + "\n")
                    with open(os.path.join(self.tmp_path, str(index) + ".obj"), 'w') as f:
                        f.write(obj)
                    shape = docNode.createElement("shape")
                    shape.setAttribute("type", "obj")
                    shape.setAttribute("id", str(index))
                    rootNode.appendChild(shape)
                    strdoc = docNode.createElement("string")
                    strdoc.setAttribute("name", "filename")
                    strdoc.setAttribute("value", str(index) + ".obj")
                    shape.appendChild(strdoc)
                    index += 1

        if len(overstory_points) > 10:
            alpha_cg = AlphaShapeCGAL(self.tmp_path, self.use_cgal)
            Utils.log("  - Segmenting overstory crowns, please wait...")
            labels = overstory_labels
            nums = area_max
            interval = 1 if nums <= 10 else int(nums/10)
            if labels is None:  # using kmeans or hexagon
                upper_left_corner = np.array([0.5 * self.scene_width, 0.5 * self.scene_height])
                label_indices = np.floor((overstory_points[:, [0, 2]] - upper_left_corner) / (-resolution)).astype(int)
                label_raster = np.zeros(areas.shape, dtype=int)
                row_idx = 0
                for (col, row) in label_indices:
                    label_raster[row, col] = 1
                    row_idx += 1

                # kmeans on image
                if self.seg_method == "kmeans":
                    nums = area_max*3
                    if self.seg_param != "default":
                        nums = int(self.seg_param)
                    interval = 1 if nums <= 10 else int(nums / 10)
                    upper_pixel = np.where(label_raster == 1)
                    upper_pixel_2d = np.vstack((upper_pixel[0], upper_pixel[1])).transpose()
                    Utils.log("  - Kmeans clustering...")
                    kmeans = KMeans(n_clusters=nums, random_state=0, init='k-means++', n_init="auto").fit(upper_pixel_2d)
                    labels = kmeans.labels_ + 1
                    label_raster = np.zeros(areas.shape, dtype=int)
                    row_idx = 0
                    for (row, col) in upper_pixel_2d:
                        label_raster[row, col] = int(labels[row_idx])
                        row_idx += 1
                    labels = label_raster[label_indices[:, 1], label_indices[:, 0]]
                elif self.seg_method == "hexagon":
                    hoxagon_size = float(self.seg_param)/resolution
                    label_raster = hexagonify(label_raster, hoxagon_size)
                    nums = label_raster.max()
                    interval = 1 if nums <= 10 else int(nums / 10)
                    labels = label_raster[label_indices[:, 1], label_indices[:, 0]]
                elif self.seg_method == "kmeans_points":  # kmeans on point cloud
                    nums = area_max * 3
                    if self.seg_param != "default":
                        nums = int(self.seg_param)
                    interval = 1 if nums <= 10 else int(nums / 10)
                    # using kmeans on point cloud directly
                    Utils.log("  - Kmeans clustering...")
                    kmeans = KMeans(n_clusters=nums, random_state=0, init='k-means++', n_init="auto").fit(overstory_points[:, [0, 2]])
                    labels = kmeans.labels_ + 1 # kmeans gives label from 0, while in raster map, 0 indicates ground
                    row_idx = 0
                    for (col, row) in label_indices:
                        label_raster[row, col] = int(labels[row_idx])
                        row_idx += 1
                # import matplotlib.pyplot as plt
                # plt.imshow(label_raster)
                # plt.show()
                areas = label_raster
                # fill holes
                rows, cols = areas.shape
                for row in range(1, rows-1):
                    for col in range(1, cols-1):
                        if areas[row, col] == 0 and (areas[row-1, col]!= 0 and areas[row-1, col] == areas[row+1, col] == areas[row, col-1] == areas[row, col+1]):
                            areas[row, col] = areas[row-1, col]
                ground_indices = np.floor((ground_points[:, [0, 2]] - upper_left_corner)/(-resolution)).astype(int)
                ground_point_labels = areas[ground_indices[:, 1], ground_indices[:, 0]]
                understory_indices = np.floor((understory_points[:, [0, 2]] - upper_left_corner)/(-resolution)).astype(int)
                understory_labels = areas[understory_indices[:, 1], understory_indices[:, 0]]

            Utils.log("  - Crown constructing...")
            for i in range(1, nums+1):
                if i % interval == 0:
                    percentage = i / nums
                    Utils.log("   - percentage: %.0f%s" % (100*percentage, "%"))
                if i == nums - 1:
                    Utils.log("   - percentage: 100%")
                subpoints = overstory_points[labels == i]  # vegetation points
                if self.pad_method == "point_number":
                    crown_points = len(subpoints)
                    crown_ground_points_num = len(ground_point_labels[ground_point_labels == i])
                    crown_understory_points_num = len(understory_labels[understory_labels == i])
                    t_points_num = crown_ground_points_num + crown_understory_points_num + crown_points
                    if t_points_num == 0:
                        T_totoal = 1
                    else:
                        T_totoal = (crown_ground_points_num + crown_understory_points_num) / t_points_num
                    crown_area = areas[areas == i]
                    project_area = len(crown_area)*resolution*resolution
                if len(subpoints) > 3:
                    meshes = []
                    pads = []
                    if self.estimation_type == "alphashape":
                        # divide 3 layers with equal height
                        subpoints_z = subpoints[:, 1]
                        min_z, max_z = min(subpoints_z), max(subpoints_z)
                        vertical_space = (max_z - min_z) / 3.0
                        acc_point_num = 0
                        for j in range(0, 3):
                            subsubpoints = subpoints[(subpoints[:, 1] > (min_z+j*vertical_space)) & (subpoints[:, 1] <= (min_z+(j+1)*vertical_space))]
                            if len(subsubpoints) > 3:
                                try:
                                    if self.use_cgal:
                                        mesh = alpha_cg.alphashape(subsubpoints, self.param)
                                    else:
                                        mesh = alphashape.alphashape(subsubpoints, alpha)
                                    if len(mesh.faces) >= 3:
                                        meshes.append(mesh)
                                        if self.pad_method == "point_number":
                                            num_subsubpoints = len(subsubpoints)
                                            total_points = crown_ground_points_num + crown_understory_points_num + acc_point_num + num_subsubpoints
                                            T = (crown_ground_points_num + crown_understory_points_num + acc_point_num)/ total_points
                                            pad_total = Utils.transmittance2lai(T) * project_area / mesh.volume
                                            pads.append(pad_total)
                                            acc_point_num += num_subsubpoints
                                        if not mesh.is_watertight:
                                            Utils.fix_normals(mesh)
                                except Exception as e:
                                    pass
                    elif self.estimation_type == "ellipsoid":
                        min_xyz = subpoints.min(axis=0)
                        max_xyz = subpoints.max(axis=0)
                        bound_box = max_xyz - min_xyz
                        center_pt = min_xyz + 0.5 * bound_box
                        mesh = trimesh.creation.icosphere()
                        mesh.apply_scale(0.5 * bound_box)
                        mesh.apply_translation(center_pt)
                        meshes.append(mesh)
                        if self.pad_method == "point_number":
                            pad_total = Utils.transmittance2lai(T_totoal) * project_area / mesh.volume
                            pads.append(pad_total)
                    elif self.estimation_type == "cone":
                        min_xyz = subpoints.min(axis=0)
                        max_xyz = subpoints.max(axis=0)
                        bound_box = max_xyz - min_xyz
                        center_pt = min_xyz + [0.5 * bound_box[0], 0, 0.5 * bound_box[2]]
                        mesh = trimesh.creation.cone(1, 1)
                        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
                        mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
                        mesh.apply_scale([0.5 * bound_box[0], bound_box[1], 0.5 * bound_box[2]])
                        mesh.apply_translation(center_pt)
                        meshes.append(mesh)
                        if self.pad_method == "point_number":
                            pad_total = Utils.transmittance2lai(T_totoal) * project_area / mesh.volume
                            pads.append(pad_total)
                    elif self.estimation_type == "cylinder":
                        min_xyz = subpoints.min(axis=0)
                        max_xyz = subpoints.max(axis=0)
                        bound_box = max_xyz - min_xyz
                        center_pt = min_xyz + 0.5 * bound_box
                        mesh = trimesh.creation.cylinder(1, 1)
                        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
                        mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
                        mesh.apply_scale(0.5 * bound_box)
                        mesh.apply_translation(center_pt)
                        meshes.append(mesh)
                        if self.pad_method == "point_number":
                            pad_total = Utils.transmittance2lai(T_totoal) * project_area / mesh.volume
                            pads.append(pad_total)
                    tmp_idx = 0
                    for mesh in meshes:
                        if len(mesh.faces) >= 3:
                            obj = trimesh.exchange.obj.export_obj(mesh)
                            crown_pad = 0
                            if self.pad_method == "point_number":
                                crown_pad = pads[tmp_idx]
                            fvol.write(str(index) + " " + str(mesh.volume) + " " + str(crown_pad) + "\n")
                            with open(os.path.join(self.tmp_path, str(index) + ".obj"), 'w') as f:
                                f.write(obj)
                            shape = docNode.createElement("shape")
                            shape.setAttribute("type", "obj")
                            shape.setAttribute("id", str(index))
                            rootNode.appendChild(shape)
                            strdoc = docNode.createElement("string")
                            strdoc.setAttribute("name", "filename")
                            strdoc.setAttribute("value", str(index) + ".obj")
                            shape.appendChild(strdoc)
                            index += 1
                            tmp_idx += 1
            alpha_cg.free_library()
            xm = docNode.toprettyxml()
            with open(os.path.join(self.tmp_path, "shapes.xml"), "w") as f:
                f.write(xm)
            fvol.close()

    def __pad_by_total_lai(self):
        tot_leaf_area = self.scene_width * self.scene_height*self.total_lai
        fout = open(os.path.join(self.tmp_path, "lad.txt"), "w")
        volume_path = os.path.join(self.tmp_path, "shapes.txt")
        names = []
        volumes = []
        total_vol = 0
        with open(volume_path) as f:
            for line in f:
                arr = line.split()
                names.append(arr[0])
                vol = float(arr[1])
                volumes.append(vol)
                total_vol += vol
        pad = tot_leaf_area/total_vol

        tot_leaf_area = 0
        for i in range(0, len(volumes)):
            fout.write(names[i] + " " + str(pad) + "\n")
            tot_leaf_area += volumes[i]*pad

        fout.close()
        Utils.log(" - Plot LAI: ", tot_leaf_area / float(self.scene_width * self.scene_height))
        return True


    def __pad_user_defined(self):
        fout = open(os.path.join(self.tmp_path, "lad.txt"), "w")
        volume_path = os.path.join(self.tmp_path, "shapes.txt")
        tot_leaf_area = 0
        with open(volume_path) as f:
            for line in f:
                arr = line.split()
                vol = float(arr[1])
                tot_leaf_area += vol * self.pad_constant
                fout.write(arr[0] + " " + str(self.pad_constant)+"\n")
        fout.close()
        Utils.log(" - Plot LAI: ", tot_leaf_area / float(self.scene_width * self.scene_height))
        return True

    def __pad_point_number(self):
        fout = open(os.path.join(self.tmp_path, "lad.txt"), "w")
        volume_path = os.path.join(self.tmp_path, "shapes.txt")

        scale_factor = 1.0
        if self.total_lai_mode == 2:
            tot_leaf_area = 0
            with open(volume_path) as f:
                for line in f:
                    arr = line.split()
                    vol = float(arr[1])
                    lad = float(arr[2])
                    if lad > 1.5:
                        lad = 1.5
                    tot_leaf_area += vol * lad
            tot_lai = tot_leaf_area / float(self.scene_width * self.scene_height)
            scale_factor = self.total_lai/tot_lai

        tot_leaf_area = 0
        with open(volume_path) as f:
            for line in f:
                arr = line.split()
                vol = float(arr[1])
                lad = float(arr[2])
                if lad > 1.5:
                    lad = 1.5
                lad = lad * scale_factor
                tot_leaf_area += vol * lad
                fout.write(arr[0] + " " + str(lad) + "\n")
        fout.close()
        Utils.log(" - Plot LAI: ", tot_leaf_area / float(self.scene_width * self.scene_height))
        return True

    def __pad_inversion_average_pad(self):
        currdir = os.path.split(os.path.realpath(__file__))[0]
        os.chdir(os.path.join(currdir, "rt"))

        # find lessrt folder
        # for devel mode
        lessrt_foler = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(currdir))),
                                    "python", "lesspy", "bin", "rt", "lessrt")
        python_folder = os.path.join(lessrt_foler, "python", "3.10")
        if os.path.exists(python_folder):
            sys.path.append(python_folder)
        # os.environ['PATH'] = lessrt_foler + os.pathsep + os.environ['PATH']
        if os.path.exists(lessrt_foler):
            os.add_dll_directory(lessrt_foler)
        # for release mode
        lessrt_foler = os.path.join(os.path.dirname(os.path.dirname(currdir)),
                                    "bin", "scripts","Lesspy", "bin", "rt", "lessrt")
        python_folder = os.path.join(lessrt_foler, "python", "3.10")
        if os.path.exists(python_folder):
            sys.path.append(python_folder)
        if os.path.exists(lessrt_foler):
            os.add_dll_directory(lessrt_foler)
        # os.environ['PATH'] = lessrt_foler + os.pathsep + os.environ['PATH']

        from mitsuba.core import Vector, Point, Ray, Thread
        from mitsuba.render import SceneHandler
        os.chdir(currdir)

        shape_path = os.path.join(self.tmp_path, "shapes.xml")
        logger = Thread.getThread().getLogger()
        logger.clearAppenders()
        fileResolver = Thread.getThread().getFileResolver()
        scenepath = os.path.dirname(shape_path).encode('utf-8')
        fileResolver.appendPath(scenepath)
        self.scene = SceneHandler.loadScene(fileResolver.resolve(shape_path.encode('utf-8')))
        self.scene.configure()
        self.scene.initialize()

        self.all_pulses = self.__parse_pulses_from_point_cloud()
        if len(self.all_pulses) == 0:
            Utils.log(" - No pulse information is recovered from point cloud.")
            Utils.log(" - Pad inversion failed. Please choose other PAD inversion method.")
            return False
        self.__calculate_incident_energy_of_each_return()

        Utils.log(" - Start ray traversal...")
        total_processed_cells = 0
        fdebug = open(os.path.join(self.tmp_path, "pt.txt"), "w")
        for pulse in self.all_pulses:
            total_processed_cells += 1
            if total_processed_cells % 20000 == 0:
                Utils.log("   - Processed Pulses: ", total_processed_cells)

            # first_echo = pulse.point_list[0]
            # first_echo = first_echo - self.ptCloud.header.min
            # first_echo[[2, 1]] = first_echo[[1, 2]]
            ray_o_original = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction)
            ray_o = ray_o_original - self.ptCloud.header.min
            ray_o[[2, 1]] = ray_o[[1, 2]]
            ray_o[0] = 0.5 * self.scene_width - ray_o[0]
            ray_o[2] = ray_o[2] - 0.5 * self.scene_height
            ray_dir = Vector(-pulse.pulse_direction[0], pulse.pulse_direction[2], pulse.pulse_direction[1])
            ray = Ray(Point(ray_o[0], ray_o[1], ray_o[2]), ray_dir, 0)
            its = self.scene.rayIntersect(ray)
            its_points = []
            if its is not None:
                tree_id = its.shape.getName()
                # Utils.log(tree_id)
                fdebug.write("%f %f %f\n" % (its.p.x, its.p.y, its.p.z))
                its_points.append((tree_id, its.p))
                while its is not None:
                    ray = Ray(its.p, ray_dir, 0)
                    ray.mint = 1e-7
                    its = self.scene.rayIntersect(ray)
                    if its is not None:
                        tree_id = its.shape.getName()
                        # Utils.log(tree_id)
                        fdebug.write("%f %f %f\n" % (its.p.x, its.p.y, its.p.z))
                        its_points.append((tree_id, its.p))
            pair_points = []
            if len(its_points) > 1:
                for i in range(len(its_points) - 1):
                    if its_points[i][0] == its_points[i + 1][0]:
                        pair_points.append(its_points[i])
                        pair_points.append(its_points[i + 1])
            if len(pair_points) > 0:
                # calculate distance from each echo to ray origin
                echo_distances = list(map(lambda xyz: np.linalg.norm(ray_o_original - xyz), pulse.point_list))
                # calculate distance of the first intersection point for each alpha shape
                next_indices = []
                its_distances = []
                its_names = []
                echo_distances.append(echo_distances[-1] + 999999)
                for i in range(0, len(pair_points)):
                    itsp = pair_points[i][1]
                    distance = np.linalg.norm(np.array([itsp.x, itsp.y, itsp.z]) - ray_o)
                    # Find the nearest non-negative distance index
                    next_index = max(range(len(echo_distances)),
                                     key=lambda i: -9999999 if (distance - echo_distances[i] > 0) else (
                                                 distance - echo_distances[i]))
                    # Utils.log(distance, echo_distances, next_index)
                    next_indices.append(next_index)
                    its_distances.append(distance)
                    its_names.append(pair_points[i][0])

                for i in range(0, len(next_indices), 2):
                    shape_name = its_names[i]
                    if next_indices[i + 1] <= len(pulse.point_list) - 1:
                        if next_indices[i] == next_indices[i + 1]:
                            # pass
                            if shape_name not in self.pad:
                                self.pad[shape_name] = [1, 0]  # (a,b), a is  number of traversal pulse, b is cumulated lad
                            else:
                                self.pad[shape_name][0] += 1
                        else:
                            path_len = its_distances[i + 1] - its_distances[i]
                            if path_len < 0.15:
                                path_len = 0.15
                            if pulse.pulse_incident_intensity[next_indices[i]] > 0:
                                T = pulse.pulse_incident_intensity[next_indices[i + 1]] / pulse.pulse_incident_intensity[
                                    next_indices[i]]
                                lad = Utils.transmittance2lad_simple(T, path_len)
                                if shape_name not in self.pad:
                                    self.pad[shape_name] = [1, lad]  # (a,b), a is  number of traversal pulse, b is cumulated lad
                                else:
                                    self.pad[shape_name][0] += 1
                                    self.pad[shape_name][1] += lad
        fdebug.close()
        volume_path = os.path.join(self.tmp_path, "shapes.txt")
        vol_dict = dict()
        with open(volume_path) as f:
            for line in f:
                arr = line.split()
                vol_dict[arr[0]] = float(arr[1])

        scale_factor = 1
        if self.total_lai_mode == 1:
            tot_leaf_area = 0
            for alpha_name in self.pad:
                lad_avg = self.pad[alpha_name][1] / float(self.pad[alpha_name][0])
                if lad_avg > 1.5:
                    lad_avg = 1.5
                vol = vol_dict[alpha_name]
                tot_leaf_area += lad_avg * vol
            tot_lai = tot_leaf_area / float(self.scene_width * self.scene_height)
            scale_factor = self.total_lai / tot_lai

        index = 0
        tot_leaf_area = 0
        fout = open(os.path.join(self.tmp_path, "lad.txt"), "w")
        for alpha_name in self.pad:
            lad_avg = self.pad[alpha_name][1]/float(self.pad[alpha_name][0]) * scale_factor
            if lad_avg > 1.5:
                lad_avg = 1.5
            self.pad[alpha_name].append(lad_avg)
            fout.write("%s %.4f\n" % (alpha_name, lad_avg))
            vol = vol_dict[alpha_name]
            tot_leaf_area += lad_avg * vol
            index += 1
        fout.close()
        Utils.log(" - Number of shapes containing leaves: ", index)
        Utils.log(" - Plot LAI: ", tot_leaf_area / float(self.scene_width * self.scene_height))
        return True

    def __pad_inversion_average_path(self):
        currdir = os.path.split(os.path.realpath(__file__))[0]
        os.chdir(os.path.join(currdir, "rt"))
        # find lessrt folder
        # for devel mode
        lessrt_foler = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(currdir))),
                                    "python", "lesspy", "bin", "rt", "lessrt")
        python_folder = os.path.join(lessrt_foler, "python", "3.10")
        sys.path.append(python_folder)
        if os.path.exists(lessrt_foler):
            os.add_dll_directory(lessrt_foler)
        # for release mode
        lessrt_foler = os.path.join(os.path.dirname(os.path.dirname(currdir)),
                                    "bin", "scripts", "Lesspy", "bin", "rt", "lessrt")
        python_folder = os.path.join(lessrt_foler, "python", "3.10")
        sys.path.append(python_folder)
        if os.path.exists(lessrt_foler):
            os.add_dll_directory(lessrt_foler)

        from mitsuba.core import Vector, Point, Ray, Thread
        from mitsuba.render import SceneHandler
        os.chdir(currdir)

        shape_path = os.path.join(self.tmp_path, "shapes.xml")
        logger = Thread.getThread().getLogger()
        logger.clearAppenders()
        fileResolver = Thread.getThread().getFileResolver()
        scenepath = os.path.dirname(shape_path).encode('utf-8')
        fileResolver.appendPath(scenepath)
        self.scene = SceneHandler.loadScene(fileResolver.resolve(shape_path.encode('utf-8')))
        self.scene.configure()
        self.scene.initialize()

        self.all_pulses = self.__parse_pulses_from_point_cloud()
        if len(self.all_pulses) == 0:
            return False
        self.__calculate_incident_energy_of_each_return()

        Utils.log(" - Start ray traversal...")
        total_processed_cells = 0
        path_lengths = dict()
        inc_out_energy = dict()  # store the incident and outgoing energy
        fdebug = open(os.path.join(self.tmp_path, "pt.txt"), "w")
        for pulse in self.all_pulses:
            total_processed_cells += 1
            if total_processed_cells % 20000 == 0:
                Utils.log("   - Processed Pulses: ", total_processed_cells)
            ray_o_original = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction)
            ray_o = ray_o_original - self.ptCloud.header.min
            ray_o[[2, 1]] = ray_o[[1, 2]]
            ray_o[0] = 0.5 * self.scene_width - ray_o[0]
            ray_o[2] = ray_o[2] - 0.5 * self.scene_height
            ray_dir = Vector(-pulse.pulse_direction[0], pulse.pulse_direction[2], pulse.pulse_direction[1])
            ray = Ray(Point(ray_o[0], ray_o[1], ray_o[2]), ray_dir, 0)
            its = self.scene.rayIntersect(ray)
            its_points = []
            if its is not None:
                tree_id = its.shape.getName()
                its_points.append((tree_id, np.array([its.p.x, its.p.y, its.p.z])))
                fdebug.write("%f %f %f\n" % (its.p.x, its.p.y, its.p.z))
                while its is not None:
                    ray = Ray(its.p, ray_dir, 0)
                    ray.mint = 1e-7
                    its = self.scene.rayIntersect(ray)
                    if its is not None:
                        tree_id = its.shape.getName()
                        its_points.append((tree_id, np.array([its.p.x, its.p.y, its.p.z])))
                        fdebug.write("%f %f %f\n" % (its.p.x, its.p.y, its.p.z))
            pair_points = []
            if len(its_points) > 1:
                for i in range(len(its_points) - 1):
                    if its_points[i][0] == its_points[i + 1][0]:
                        pair_points.append(its_points[i])
                        pair_points.append(its_points[i + 1])
            if len(pair_points) > 0:
                for i in range(0, len(pair_points), 2):
                    path_len = np.linalg.norm(pair_points[i + 1][1] - pair_points[i][1])
                    if path_len<0.5:
                        path_len = 0.5
                    obj_name = pair_points[i][0]
                    if obj_name not in path_lengths:
                        path_lengths[obj_name] = [path_len]
                    else:
                        path_lengths[obj_name].append(path_len)
                # calculate distance from each echo to ray origin
                echo_distances = list(map(lambda xyz: np.linalg.norm(ray_o_original - xyz), pulse.point_list))
                # calculate distance of the first intersection point for each alpha shape
                next_indices = []
                its_distances = []
                its_names = []
                echo_distances.append(echo_distances[-1] + 999999)
                for i in range(0, len(pair_points)):
                    itsp = pair_points[i][1]
                    distance = np.linalg.norm(itsp - ray_o)
                    # Find the nearest non-negative distance index
                    next_index = max(range(len(echo_distances)),
                                     key=lambda i: -9999999 if (distance - echo_distances[i] > 0) else (
                                             distance - echo_distances[i]))
                    # Utils.log(distance, echo_distances, next_index)
                    next_indices.append(next_index)
                    its_distances.append(distance)
                    its_names.append(pair_points[i][0])

                for i in range(0, len(next_indices), 2):
                    shape_name = its_names[i]
                    if next_indices[i] == next_indices[i + 1]:
                        if next_indices[i + 1] <= len(pulse.point_list) - 1:
                            if shape_name not in inc_out_energy:
                                inc_out_energy[shape_name] = np.array([pulse.pulse_incident_intensity[next_indices[i]],
                                                                       pulse.pulse_incident_intensity[next_indices[i]]],
                                                                      dtype=float)
                            else:
                                inc_out_energy[shape_name] += np.array([pulse.pulse_incident_intensity[next_indices[i]],
                                                                        pulse.pulse_incident_intensity[
                                                                            next_indices[i]]], dtype=float)
                    else:
                        if next_indices[i] <= len(pulse.point_list) - 1:
                            if shape_name not in inc_out_energy:
                                inc_out_energy[shape_name] = np.array([pulse.pulse_incident_intensity[next_indices[i]],
                                                                       0], dtype=float)
                            else:
                                inc_out_energy[shape_name] += np.array([pulse.pulse_incident_intensity[next_indices[i]],
                                                                        0], dtype=float)
                        if next_indices[i + 1] <= len(pulse.point_list) - 1:
                            if shape_name not in inc_out_energy:
                                inc_out_energy[shape_name] = np.array([0,
                                                                       pulse.pulse_incident_intensity[
                                                                           next_indices[i + 1]]])
                            else:
                                inc_out_energy[shape_name] += np.array([0,
                                                                        pulse.pulse_incident_intensity[
                                                                            next_indices[i + 1]]])
        volume_path = os.path.join(self.tmp_path, "shapes.txt")
        vol_dict = dict()
        with open(volume_path) as f:
            for line in f:
                arr = line.split()
                vol_dict[arr[0]] = float(arr[1])
        tot_leaf_area = 0
        index = 0
        for obj_name in path_lengths:
            avg_length = sum(path_lengths[obj_name])/float(len(path_lengths[obj_name]))
            if obj_name in inc_out_energy:
                if inc_out_energy[obj_name][0] > 0:
                    trans = inc_out_energy[obj_name][1] / float(inc_out_energy[obj_name][0])
                    if 0 < trans < 1 and avg_length > 0:
                        lad = Utils.transmittance2lad_simple(trans, avg_length)
                        self.pad[obj_name] = lad
                        leaf_area = vol_dict[obj_name] * lad
                        tot_leaf_area += leaf_area
                        index += 1
        Utils.log(" - Number of shapes containing leaves: ", index)
        Utils.log(" - Plot LAI: ", tot_leaf_area / float(self.scene_width * self.scene_height))
        return True

    def __create_less_simulation(self):
        pylesssdk_path = os.path.join(os.path.dirname(os.path.split(os.path.realpath(__file__))[0]), "pyLessSDK")
        sys.path.append(pylesssdk_path)
        from SimulationHelper import SimulationHelper
        from Simulation import Simulation
        from SceneObjects import SceneObject
        from OpticalProperty import OpticalItem
        import Terrain
        Utils.log(" - Creating LESS simulation...")
        sim_helper = SimulationHelper(self.less_root_dir)  # 创建SimulationHelper，参数为LESS的安装根目录
        sim_helper.create_new_sim(self.less_sim_path)  # 新建模拟工程
        sim = Simulation(self.less_sim_path, sim_helper)  # 初始化Simulation对象
        sim.read_sim_project()  # 读取模拟工程的内容
        scene = sim.get_scene()  # 得到Scene
        landscape = scene.get_landscape()  # 得到LandScape对象
        landscape.clear_landscape_elements()  # # 清除工程已有的场景元素（重复运行程序时，必须执行）
        landscape.convert_obj_to_binary(False)

        sensor = scene.get_sensor()
        sensor.set_sample_per_pixel(512)
        sensor.set_image_width(400)
        sensor.set_image_height(400)
        sensor.set_sub_region_width(self.scene_width)
        sensor.set_sub_region_height(self.scene_height)
        sensor.set_spectral_bands("450:1,550:1,650:1,850:1")
        ill_obj = scene.get_illumination()
        ill_obj.set_ats_percentage("0,0,0,0")

        terrain = landscape.get_terrain()
        terrain.set_extent_width(self.scene_width)
        terrain.set_extent_height(self.scene_height)
        terrain.set_terrain_type(Terrain.TERRAIN_TYPE.RASTER)
        terrain.set_terrain_file(os.path.join(self.tmp_path, "dem"))

        dem_data, x_size, y_size = Utils.read_img_to_arr_no_transform(os.path.join(self.tmp_path, "dem"))
        center_relative_altitude = dem_data[int(0.5 * y_size)][int(0.5 * x_size)] - dem_data.min()

        op_item01 = OpticalItem(self.leaf_op, "0.0454,0.1012,0.0389,0.4747;0.0454,0.1012,0.0389,0.4747;0.0005,0.1067,0.0187,0.4843")  # LESS初始时为两个波段
        landscape.add_op_item(op_item01)  # 添加到场景库

        f = open(os.path.join(self.tmp_path, "lad.txt"))
        if self.leaf_as_facet:
            Utils.log(" - Generating facet leaves...")
        obj_index = 0
        all_lines = f.readlines()
        tot = len(all_lines)
        for line in all_lines:
            arr = line.split()
            obj_name = "OBJ"+arr[0]
            obj_path = os.path.join(self.tmp_path, arr[0]+".obj")
            leaf_density = float(arr[1])
            if not self.leaf_as_facet:
                obj = SceneObject(obj_name)  # 定义一个场景物体，名叫Tree01
                obj.add_component_from_file(obj_path, self.leaf_op,
                                            is_turbid=True, leaf_density=leaf_density, lad=self.leaf_angle_dist, hotspot_factor=0.1)
                landscape.add_object(obj)
                landscape.place_object(obj_name, x=0.5 * self.scene_width, y=0.5 * self.scene_height,
                                       z=-center_relative_altitude)
            else:
                if obj_index % 100 == 0:
                    Utils.log(" - (%d/%d)tree objects have been created." % (obj_index+1, tot))
                obj_mesh = trimesh.exchange.load.load(obj_path)
                bounds_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
                bounds_vol = bounds_size[0] * bounds_size[1] * bounds_size[2]
                num_leaves_boundbox = int(leaf_density*bounds_vol / self.single_leaf_area)
                leaf_pos = trimesh.sample.volume_mesh(obj_mesh, num_leaves_boundbox)
                num_leaves = leaf_pos.shape[0]
                real_num_leaves = int(leaf_density*obj_mesh.volume / self.single_leaf_area)
                if num_leaves > 0:
                    # print(num_leaves, real_num_leaves)
                    leaves_generator = LeafAsFacetGenerator(self.leaf_angle_dist, num_leaves, leaf_pos, self.single_leaf_area)
                    leaves_obj_path = os.path.join(self.tmp_path, arr[0] + "_leaves.obj")
                    leaves_generator.generate_leaves(leaves_obj_path)
                    obj = SceneObject(obj_name)  # 定义一个场景物体，名叫Tree01
                    obj.add_component_from_file(leaves_obj_path, self.leaf_op)
                    landscape.add_object(obj)
                    landscape.place_object(obj_name, x=0.5*self.scene_width, y=0.5*self.scene_height, z=-center_relative_altitude)
            obj_index += 1
        f.close()
        sim.prepare_for_ui()
        sim.save_sim_project()

    def __parse_pulses_from_point_cloud(self):
        Utils.log(" - Parse pulse information from point cloud...")
        sourceID_unique = set(self.ptCloud.pt_src_id)
        Utils.log("  - Total number of points: ", self.ptCloud.header.point_count)
        final_pulses = []
        error_pulses = []
        xyz = self.ptCloud.xyz
        for srcID in sourceID_unique:
            xyz_total = xyz[self.ptCloud.pt_src_id == srcID]
            number_of_returns = self.ptCloud.num_returns[self.ptCloud.pt_src_id == srcID]
            return_number = self.ptCloud.return_num[self.ptCloud.pt_src_id == srcID]
            gps_time = self.ptCloud.gps_time[self.ptCloud.pt_src_id == srcID]
            if self.ptCloud.point_format.id == 6:
                scan_angle_rank = self.ptCloud.scan_angle[self.ptCloud.pt_src_id == srcID]
            else:
                scan_angle_rank = self.ptCloud.scan_angle_rank[self.ptCloud.pt_src_id == srcID]
            classification = self.ptCloud.classification[self.ptCloud.pt_src_id == srcID]
            intensity = self.ptCloud.intensity[self.ptCloud.pt_src_id == srcID]
            D = defaultdict(list)
            for i, item in enumerate(gps_time):
                D[item].append(i)
            D = {k: v for k, v in list(D.items())}
            for key in D:
                pulse = Pulse()
                for id in D[key]:
                    pulse.point_list.append(xyz_total[id])
                    pulse.return_number_list.append(return_number[id])
                    pulse.gps_time_list.append(gps_time[id])
                    pulse.classification_list.append(classification[id])
                    pulse.intensity_list.append(intensity[id])
                    pulse.scan_angle_list.append(scan_angle_rank[id])
                    pulse.number_of_return_list.append(number_of_returns[id])
                    pulse.source_id = srcID
                # validate pulse
                # return number: it should not have duplicated values.
                if len(set(pulse.return_number_list)) != len(pulse.return_number_list):
                    error_pulses.append(pulse)
                    continue
                # number of return: it should be the same
                if len(set(pulse.number_of_return_list)) != 1:
                    error_pulses.append(pulse)
                    continue
                # number of return should equal to return number
                if pulse.number_of_return_list[0] != len(pulse.return_number_list):
                    error_pulses.append(pulse)
                    continue
                # scan angle
                if len(set(pulse.scan_angle_list)) != 1:
                    error_pulses.append(pulse)
                    continue
                final_pulses.append(pulse)
        errorPoints = count_points_of_pulse_list(error_pulses)
        Utils.log("  - Total number of points from incomplete pulses: ", errorPoints, "->",
              "%.2f" % (100 * errorPoints / len(xyz)), "%")

        # sort the points in each pulse according to return number
        for pulse in final_pulses:
            pulse.point_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.point_list))]
            pulse.scan_angle_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.scan_angle_list))]
            pulse.classification_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.classification_list))]
            pulse.intensity_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.intensity_list))]
            pulse.return_number_list = sorted(pulse.return_number_list)

        # determine the type of pulse according to classification
        num_vg, num_g, num_v = 0, 0, 0
        for pulse in final_pulses:
            if 1 in pulse.classification_list and 2 in pulse.classification_list:
                pulse.pulse_type = PulseType.VEG_GROUND
                num_vg += 1
            elif all([x == 2 for x in pulse.classification_list]):
                pulse.pulse_type = PulseType.PURE_GROUND
                num_g += 1
            else:
                pulse.pulse_type = PulseType.PURE_VEG
                num_v += 1

        Utils.log("  - Total number of Pulses: ", len(final_pulses))
        if len(final_pulses) > 0:
            Utils.log("    - Pure-Vegetation Pulse: ", num_v, " -> %.2f" % (100 * num_v / len(final_pulses)), "%")
            Utils.log("    - Pure-Ground Pulse: ", num_g, " -> %.2f" % (100 * num_g / len(final_pulses)), "%")
            Utils.log("    - Vegetation-ground Pulse: ", num_vg, " -> %.2f" % (100 * num_vg / len(final_pulses)), "%")

        # determine the pulse direction
        scan_angle_direction_dic = dict()
        for sourceID in sourceID_unique:
            scan_angle_direction_dic[sourceID] = dict()
        for pulse in final_pulses:
            point_num = pulse.get_point_num()
            # directly get pulse direction from the coordinates of the points
            # for one-return pulses, using nearest multiple-return pulses
            if point_num > 1:
                p_dir = np.array(pulse.point_list[point_num - 1] - pulse.point_list[0])
                p_dir /= np.linalg.norm(p_dir)
                pulse.pulse_direction = p_dir
                scan_angle = pulse.get_scann_angle_rank()
                source_id = pulse.source_id
                if scan_angle in scan_angle_direction_dic[source_id]:
                    scan_angle_direction_dic[source_id][scan_angle].append(pulse.pulse_direction)
                else:
                    scan_angle_direction_dic[source_id][scan_angle] = [pulse.pulse_direction]
        for sourceID in sourceID_unique:
            for scan_angle in scan_angle_direction_dic[sourceID]:
                scan_angle_direction_dic[sourceID][scan_angle] = sum(scan_angle_direction_dic[sourceID][scan_angle]) \
                                                       / float(len(scan_angle_direction_dic[sourceID][scan_angle]))
            # Utils.log(scan_angle, scan_angle_direction_dic[scan_angle])
        pulse_with_no_direction = 0
        for pulse in final_pulses:
            if pulse.get_point_num() == 1:
                if pulse.get_scann_angle_rank() in scan_angle_direction_dic[pulse.source_id]:
                    pulse.pulse_direction = scan_angle_direction_dic[pulse.source_id][pulse.get_scann_angle_rank()]
                else:
                    pulse.pulse_direction = np.array([0, 0, -1], dtype=float)
                    pulse_with_no_direction += 1
        if len(final_pulses)>0:
            Utils.log("  - Pulse with no direction: ", pulse_with_no_direction,
                  " -> %.2f" % (100 * pulse_with_no_direction / len(final_pulses)), "%")
        return final_pulses

    def __calculate_incident_energy_of_each_return(self):
        Utils.log(" - Determine incident intensity of returns...")
        # find pure ground pulse to build nearest search data structure
        # pulse_type == PURE_GROUND
        pure_ground_points_array = []
        pure_ground_intensity_array = []
        for pulse in self.all_pulses:
            if pulse.pulse_type == PulseType.PURE_GROUND:
                pure_ground_points_array += pulse.point_list
                pure_ground_intensity_array += pulse.intensity_list
        pure_ground_points_array = np.array(pure_ground_points_array) #  - np.array( [self.bb.min_x, self.bb.min_y, self.bb.min_z])
        # using x y of pure ground point to build a nearest search tree
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pure_ground_points_array[:, 0:2])
        number_of_invalided_ground_veg_points = 0

        dem_path = os.path.join(self.tmp_path, "dem")
        exr_path = os.path.join(self.tmp_path, "dem.exr")
        terrain_path = os.path.join(self.tmp_path, "terrain.xml")
        Utils.convert_dem_to_mip_monochromatic_chanel(dem_path, exr_path)
        Utils.write_terrain_xml(terrain_path, self.scene_width, self.scene_height)
        # currdir = os.path.split(os.path.realpath(__file__))[0]
        # os.chdir(os.path.join(currdir, "rt"))
        # sys.path.append(os.path.join(currdir, "rt", "python", "3.6"))
        # os.environ['PATH'] = os.path.join(currdir, "rt") + os.pathsep + os.environ['PATH']
        from mitsuba.core import Vector, Point, Ray, Thread
        from mitsuba.render import SceneHandler
        # os.chdir(currdir)
        logger = Thread.getThread().getLogger()
        logger.clearAppenders()
        fileResolver = Thread.getThread().getFileResolver()
        scenepath = os.path.dirname(terrain_path).encode('utf-8')
        fileResolver.appendPath(scenepath)
        self.terr_scene = SceneHandler.loadScene(fileResolver.resolve(terrain_path.encode('utf-8')))
        self.terr_scene.configure()
        self.terr_scene.initialize()

        for pulse in self.all_pulses:
            pulse_intensity_list = list(map(lambda x: float(x), pulse.intensity_list))
            point_num = pulse.get_point_num()

            # calculate incident energy at return position
            # for vegetation-ground pulse, use the nearest ground pulse to correct ground reflectance
            incident_energy_list = [0 for i in range(0, point_num)]
            if pulse.pulse_type == PulseType.VEG_GROUND:
                # distances, indices = nbrs.kneighbors([pulse_point_cell_list[point_num - 1][0:2]])
                terr_xy = pulse.point_list[point_num - 1][0:2]
                ray_o = pulse.point_list[0] - self.ptCloud.header.min
                ray_dir = Vector(pulse.pulse_direction[0], pulse.pulse_direction[1], pulse.pulse_direction[2])
                ray = Ray(Point(ray_o[0], ray_o[1], ray_o[2]), ray_dir, 0)
                its = self.terr_scene.rayIntersect(ray)
                if its is not None:
                    terr_xy = np.array([its.p.x, its.p.y]) + self.ptCloud.header.min[0:2]
                distances, indices = nbrs.kneighbors([terr_xy])
                qgstart = pure_ground_intensity_array[indices[0][0]]  # near_pure_ground_intensity
                qg = pulse_intensity_list[point_num - 1]  # last_ground_intensity

                if qgstart <= qg:
                    number_of_invalided_ground_veg_points += 1
                    for i in range(0, point_num):
                        incident_energy_list[i] = sum(pulse_intensity_list[i:])

                else:
                    for i in range(0, point_num):
                        incident_energy_list[i] = qg * sum(pulse_intensity_list[0:i]) + \
                                                  qgstart * sum(pulse_intensity_list[i: point_num - 1])
                        incident_energy_list[i] /= (qgstart - qg)
            if pulse.pulse_type == PulseType.PURE_VEG:
                for i in range(0, point_num):
                    incident_energy_list[i] = sum(pulse_intensity_list[i:])
            pulse.pulse_incident_intensity = incident_energy_list

        Utils.log("  - Invalid Ground Points: ", number_of_invalided_ground_veg_points, " -> ",
              "%.2f" % (100 * number_of_invalided_ground_veg_points / count_points_of_pulse_list(self.all_pulses)), "%")