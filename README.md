# DCCVT

Code entry point : DCCVT_experiments.py 

## Installation

### Conda

Create conda env from env.yml

### Virtual env

Create a virtual env. Ensure to have all submodule clone:
```git submodule update --init --recursive``

Open3D:
```
cd 3rdparty/open3d
git apply ../open3d.patch
mkdir build
cd build
cmake -GNinja ..
cmake --build . --config Release --target install-pip-package
```

diffvoronoi module is from for Delaunay Parallel 3d :
https://github.com/BrunoLevy/geogram/wiki/Delaunay3D
 
