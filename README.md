# ALS2LAI3D
This code estimates 3D leaf area density ($m^2/m^3$) from airborne or UAV LiDAR point cloud and reconstructs 3D scenes for radiative transfer simulations. Several steps are automatically executed, including:
* Ground filtering using [CSF](https://github.com/jianboqi/CSF)
* Understory and overstory speration
* Tree crown segmentation
* Crown boundary(alphashape, voxel, ellipsoid, cone) creation
* Leaf area density estimation using pulse tracing, point number, or user-defined LAD, scene LAI.
* 3D scene creation for 3D radiative transfer models (LESS currently)


