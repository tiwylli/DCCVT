# DCCVT

## Installation

### Conda

Create conda env from env.yml

### Virtual env

#### Cluster

```
module load cuda python-build-bundle/2025a clang
```

Virtual env setup with no index to use the optimized verison
```
pip install --no-index torch
pip install --no-index matplotlib
pip install --no-index pybind11
pip install polyscope
pip install --no-index scipy scikit-image
pip install --no-index cython
pip install usd-core flask tornado comm ipyevents ipycanvas jupyter_client
pip install open3d
```

#### Local machine

Install the pip listed below. Do not need to perform the module command line.

#### Compile and install other dependencies


Create a virtual env. Ensure to have all submodule clone:
```git submodule update --init --recursive``

Most of the dependency can be installed via pip (see below the list of packages). Note that some package do not need to be compiled if you are using not a too advanced python intepreteur version.

Pytorch3d:
```
cd 3rdparty/pytorch3d
python setup.py build
python setup.py install
```

gDel3D:
```
cd 3rdparty/gDel3D/python_bindings
mkdir build
cd build
cp pygdel3d.cpython-311-x86_64-linux-gnu.so <..INTOVENVSITEPACKAGE..>
```

Kaolin
```
cd 3rdparty/kaolin
git apply ../kaolin.patch
python setup.py build
python setup.py install
```

diffvoronoi module is from for Delaunay Parallel 3d :
https://github.com/BrunoLevy/geogram/wiki/Delaunay3D

#### Deprecated (or broken)

USD -- can be installed with pip
Open3D -- can be installed with pip. Old instructions as backup
```
cd 3rdparty/open3d
git apply ../open3d.patch
mkdir build
cd build
cmake -GNinja ..
cmake --build . --config Release --target install-pip-package
```

