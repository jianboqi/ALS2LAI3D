# coding: utf-8
# This scripts merge las files into single one, with sourceID specified.
# usage:
# python laspyMerge -i ./*.las -o merged.las

import argparse
import laspy
import numpy as np
import glob

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-i", nargs='*', help="Input las file.", required=True)
    parse.add_argument("-o", help="Output las file.", required=True)
    args = parse.parse_args()
    input_las_files = []
    output_las_file = args.o
    for input_las_file in args.i:
        input_las_files += glob.glob(input_las_file)

    hdr = laspy.LasHeader(version="1.4", point_format=1)
    outFile = laspy.LasData(hdr)

    allx = np.array([])
    ally = np.array([])
    allz = np.array([])
    intensity = np.array([])
    return_number = np.array([], dtype=np.uint8)
    number_of_return = np.array([], dtype=np.uint8)
    gps_time = np.array([])
    classification = np.array([], dtype=np.uint8)
    scan_angle = np.array([], dtype=np.int8)
    sourceID = np.array([], dtype=np.uint16)
    edge_point = np.array([], dtype=np.uint8)

    sourceIndex = 0
    for input_las_file in input_las_files:
        print(" - **********************************")
        print(" - Processing las file: ", input_las_file)
        inFile = laspy.read(input_las_file)
        allx = np.concatenate((allx, inFile.x))
        ally = np.concatenate((ally, inFile.y))
        allz = np.concatenate((allz, inFile.z))
        intensity = np.concatenate((intensity, inFile.intensity))
        return_number = np.concatenate((return_number, inFile.return_num))
        number_of_return = np.concatenate((number_of_return, inFile.num_returns))
        gps_time = np.concatenate((gps_time, inFile.gps_time))
        classification = np.concatenate((classification, inFile.classification))
        scan_angle = np.concatenate((scan_angle, inFile.scan_angle_rank))
        sourceID = np.concatenate((sourceID, inFile.pt_src_id+sourceIndex))
        edge_point = np.concatenate((edge_point, inFile.edge_flight_line))
        sourceIndex += 1
    xmin = np.floor(np.min(allx))
    ymin = np.floor(np.min(ally))
    zmin = np.floor(np.min(allz))

    outFile.header.offset = [xmin, ymin, zmin]
    outFile.header.scale = [0.001, 0.001, 0.001]

    outFile.x = allx
    outFile.y = ally
    outFile.z = allz
    outFile.pt_src_id = sourceID
    outFile.intensity = intensity
    outFile.return_num = return_number
    outFile.num_returns = number_of_return
    outFile.gps_time = gps_time
    outFile.classification = classification
    outFile.scan_angle_rank = scan_angle
    outFile.edge_flight_line = edge_point
    outFile.write(output_las_file)

    # outFile.close()
