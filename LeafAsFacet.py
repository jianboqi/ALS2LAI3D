# coding: utf-8
import numpy as np
from scipy.linalg import expm, norm

class LAD:
    UNIFORM = "Uniform"
    SPHERICAL = "Spherical"
    ERECTOPHILE = "Erectophile"
    EXTREMOPHILE = "Extremophile"
    PLANOPHILE = "Planophile"
    PLAGIOPHILE = "Plagiophile"

class LeafShape:
    SQUARE = "Square"
    DISK = "Disk"


class LeafAsFacetGenerator(object):
    def __init__(self, leaf_angle_dist, num_leaves, leaf_positions, single_leaf_area, leaf_shape=LeafShape.SQUARE):
        self.leaf_angle_dist = leaf_angle_dist
        self._num_of_leaves = num_leaves
        self.all_pos = leaf_positions
        self.single_leaf_area = single_leaf_area
        self.leaf_shape = leaf_shape
        self.leaf_num_triangles = 12

    def _calculate_leaf_length(self):
        if self.leaf_shape == LeafShape.SQUARE:
            return np.sqrt(self.single_leaf_area)
        if self.leaf_shape == LeafShape.DISK:
            theta = np.pi * 2 / self.leaf_num_triangles
            tri_leaf = self.single_leaf_area / self.leaf_num_triangles
            leaf_radius = np.sqrt(2 * tri_leaf / np.sin(theta))
            return leaf_radius


    def _generate_leaf_normal(self):
        '''
        Sphrical: 球形状
        Uniform:统一型
        Planophile：平面型
        Erectophile：竖直型
        Plagiophile：倾斜型
        Extremophile：极端型
        '''
        all_normals = []
        if self.leaf_angle_dist == LAD.SPHERICAL:
            for i in range(self._num_of_leaves):
                phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                theta = np.arccos(np.random.rand())  # randomly zenith angle
                rx = -np.sin(theta) * np.cos(phi)
                rz = np.sin(theta) * np.sin(phi)
                ry = np.cos(theta)
                all_normals.append([rx, ry, rz])
            # f = open(r"D:\LESS\simulations\LESSMedium\SimpleHorizontalLayerRAMI\HOM03_ERE_scene.def")
            # for line in f:
            #     arr = list(map(lambda x: float(x), line.split()))
            #     all_normals.append([-arr[4], arr[6], arr[5]])
        if self.leaf_angle_dist == LAD.UNIFORM:
            for i in range(self._num_of_leaves):
                phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                theta = np.random.rand() * np.pi * 0.5  # randomly zenith angle
                rx = -np.sin(theta) * np.cos(phi)
                rz = np.sin(theta) * np.sin(phi)
                ry = np.cos(theta)
                all_normals.append([rx, ry, rz])
        if self.leaf_angle_dist == LAD.PLANOPHILE:
            while True:
                theta = np.random.rand() * np.pi * 0.5
                y = (2 + 2 * np.cos(2 * theta)) / np.pi
                rnd_y = np.random.rand() * 4 / np.pi
                if rnd_y <= y:  # accept
                    phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                    rx = -np.sin(theta) * np.cos(phi)
                    rz = np.sin(theta) * np.sin(phi)
                    ry = np.cos(theta)
                    all_normals.append([rx, ry, rz])
                    if len(all_normals) == self._num_of_leaves:
                        break
        if self.leaf_angle_dist == LAD.ERECTOPHILE:  # 竖直型
            while True:
                theta = np.random.rand() * np.pi * 0.5
                y = (2 - 2 * np.cos(2 * theta)) / np.pi
                rnd_y = np.random.rand() * 4 / np.pi
                if rnd_y <= y:  # accept
                    phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                    rx = -np.sin(theta) * np.cos(phi)
                    rz = np.sin(theta) * np.sin(phi)
                    ry = np.cos(theta)
                    all_normals.append([rx, ry, rz])
                    if len(all_normals) == self._num_of_leaves:
                        break
        if self.leaf_angle_dist == LAD.PLAGIOPHILE:  # 倾斜型
            while True:
                theta = np.random.rand() * np.pi * 0.5
                y = (2 - 2 * np.cos(4 * theta)) / np.pi
                rnd_y = np.random.rand() * 4 / np.pi
                if rnd_y <= y:  # accept
                    phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                    rx = -np.sin(theta) * np.cos(phi)
                    rz = np.sin(theta) * np.sin(phi)
                    ry = np.cos(theta)
                    all_normals.append([rx, ry, rz])
                    if len(all_normals) == self._num_of_leaves:
                        break
        if self.leaf_angle_dist == LAD.EXTREMOPHILE:  # 极端
            while True:
                theta = np.random.rand() * np.pi * 0.5
                y = (2 + 2 * np.cos(4 * theta)) / np.pi
                rnd_y = np.random.rand() * 4 / np.pi
                if rnd_y <= y:  # accept
                    phi = np.random.rand() * np.pi * 2  # randomly azimuth angle
                    rx = -np.sin(theta) * np.cos(phi)
                    rz = np.sin(theta) * np.sin(phi)
                    ry = np.cos(theta)
                    all_normals.append([rx, ry, rz])
                    if len(all_normals) == self._num_of_leaves:
                        break
        return all_normals

    @staticmethod
    def _M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def _generate_single_leaf(self, leaf_pos, leaf_normal, leaf_length):
        leaf_side_length = leaf_length
        # generate leaf according to normal
        if self.leaf_shape == LeafShape.SQUARE:
            # First, find a arbitrary vector which is not parallel with normal
            # and do cross product, the resulting vector is in leaf plane
            arv = np.array([0, 0, 1])
            v = np.cross(leaf_normal, arv)
            v = v / np.linalg.norm(v)
            p1 = leaf_side_length * np.sqrt(2) * 0.5 * v + leaf_pos
            v = np.cross(leaf_normal, v)
            p2 = leaf_side_length * np.sqrt(2) * 0.5 * v + leaf_pos
            v = np.cross(leaf_normal, v)
            p3 = leaf_side_length * np.sqrt(2) * 0.5 * v + leaf_pos
            v = np.cross(leaf_normal, v)
            p4 = leaf_side_length * np.sqrt(2) * 0.5 * v + leaf_pos
            return [p1, p2, p3, p4]
        elif self.leaf_shape == LeafShape.DISK:  # leaf_side_length is leaf radius
            # First, find a arbitrary vector which is not parallel with normal
            # and do cross product, the resulting vector is in leaf plane
            pts = []
            rot_rad = np.pi * 2 / self.leaf_num_triangles
            arv = np.array([0, 0, 1])
            v0 = np.cross(leaf_normal, arv)
            v0 = v0 / np.linalg.norm(v0)
            p1 = leaf_side_length * v0 + leaf_pos  # 第一个点
            pts.append(p1)
            for i in range(0, self.leaf_num_triangles - 1):
                m0 = LeafAsFacetGenerator._M(np.array(leaf_normal), rot_rad * (i + 1))
                newv = np.dot(m0, v0)
                newv = newv / np.linalg.norm(newv)
                newp = leaf_side_length * newv + leaf_pos  # 第一个点
                pts.append(newp)
            return pts
        else:
            print("Leaf shape is not supported.")

    def generate_leaves(self, dist_obj):
        all_normal = self._generate_leaf_normal()
        leaf_length = self._calculate_leaf_length()
        f_out = open(dist_obj, "w")
        f_out.write("g leaves\n")
        trunk_start_index = 0
        for i in range(self._num_of_leaves):
            leaf_pos = self.all_pos[i]
            leaf_normal = all_normal[i]
            points = self._generate_single_leaf(leaf_pos, leaf_normal, leaf_length)
            if i % 10000 == 0:
                f_out.flush()
            if self.leaf_shape == LeafShape.SQUARE:
                p1, p2, p3, p4 = points
                f_out.write("v %.4f %.4f %.4f\n" % (p1[0], p1[1], p1[2]))
                f_out.write("v %.4f %.4f %.4f\n" % (p2[0], p2[1], p2[2]))
                f_out.write("v %.4f %.4f %.4f\n" % (p3[0], p3[1], p3[2]))
                f_out.write("v %.4f %.4f %.4f\n" % (p4[0], p4[1], p4[2]))
                f_out.write(
                    "f %d %d %d %d\n" % (trunk_start_index + 1, trunk_start_index + 2, trunk_start_index + 3, trunk_start_index + 4))
                trunk_start_index += 4
            if self.leaf_shape == LeafShape.DISK:
                points.append(leaf_pos)
                tot_pt_each_leaf = len(points)
                for j in range(0, len(points)):
                    f_out.write("v %.4f %.4f %.4f\n" % (points[j][0], points[j][1], points[j][2]))
                for j in range(0, self.leaf_num_triangles - 1):
                    f_out.write(
                        "f %d %d %d\n" % (trunk_start_index + j + 1,
                                          trunk_start_index + j + 2,
                                          trunk_start_index + tot_pt_each_leaf))
                f_out.write(
                    "f %d %d %d\n" % (trunk_start_index + self.leaf_num_triangles,
                                      trunk_start_index + 1,
                                      trunk_start_index + tot_pt_each_leaf))
                trunk_start_index += tot_pt_each_leaf
        f_out.close()