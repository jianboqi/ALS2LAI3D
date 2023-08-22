# ALS2LAI3D
This code estimates 3D leaf area density ($m^2/m^3$) from airborne or UAV LiDAR point cloud and reconstructs 3D scenes for radiative transfer simulations. Several steps are automatically executed, including:
* Ground filtering using [CSF](https://github.com/jianboqi/CSF)
* Understory and overstory speration
* Tree crown segmentation
* Crown boundary(alphashape, voxel, ellipsoid, cone) creation
* Leaf area density estimation using pulse tracing, point number, or user-defined LAD, scene LAI.
* 3D scene creation for 3D radiative transfer models (LESS currently)

There are two approaches to use this code, one is directly from LESS, which provides a user interface, the other way is to directly use the python code provided within this repo.
## Approach 1 (GUI)
please go to [Download LESS](http://lessrt.org/download-less/) to download the latest version of LESS. LESS is a ray-tracing based 3D radiative transfer model, which provides both graphic user interface (GUI) as well as Python SDK to perform 3D radiative transfer simulation over detailed 3D scenes (e.g., forests and city buildings). LESS can simulate bidirectional reflectance factor, multispectral/hyperspectral images, LiDAR waveforms/point cloud, thermal infared images, FPAR, etc. We recommend to install LESS in a path without spaces to avoid some possible issues during simulation, e.g., D:\LESS. (Note: recent LESS versions only support Windows)

Click 【Tools】【3D Object Creation】【3D Forest from LiDAR (ALS)】,then you will get a GUI dialog as follows:
![微信截图_20230822180051(1)](https://github.com/jianboqi/ALS2LAI3D/assets/1770654/37b10585-b6d4-4815-8963-62ae9f8dbf3f)

Input the ALS data or UAV data as *.las files. Then you can configure parameters, including crown types to estimate leaf area density, and estimation method.
Specify a out directory, after the execution, you will get a LESS simulation folder under the out diectory, you can use LESS GUI to open the simulation to 
do visualization or perform radiative transfer simulations. If you just want to use the estimated paramters, you can refer to the `_tmp_` foler, which constains all the obj files that represent 
the crown boudaries. `lad.txt` contains the leaf area density of each shapes, and `shapes.txt` contains the volume of each shape (the second column), the first column of `lad.txt` and `shapes.txt` is the name of the generated obj files within the folder.

## Approach 2 (Python Code)
You can directly refer to the `als2less.py` within the code. A more easier way to interprete the parameter setting is also to use the GUI, after an excution, you will find a command line in the output section of the LESS GUI. Then you can copy the code, e.g., 
```
D:\LESS\app\bin\python\python  D:\LESS\app\Python_script\ALS2LESS\als2less.py  -i  C:\Users\DELL\Desktop\ALS_sample.las  -o  C:\Users\DELL\Desktop  -sim_name  sim_proj01  -seg_method  watershed  -seg_param  4.5  -type  alphashape  -param  5  -include_understory  false  -pad_method  pulse_tracing  -pad_constant  0.8  -total_lai  3.0  -understory_height  auto  -leaf_angle_dist  Spherical  -leaf_op  leaf_op_name01  -leaf_as_facet  false  -single_leaf_area  0.01
```

>Please note that if you do not use the python interpreter installed with LESS (e.g., D:\LESS\app\bin\python\python.exe), you will need to install sklearn, CSF, laspy for your own python.


