# DCCVT

## Installation

### Conda

Create conda environment from env.yml:
```bash
conda env create -f env.yml
```

### Virtual Environment

#### Cluster

Load required modules:
```bash
module load cuda python-build-bundle/2025a clang
```

Virtual environment setup with no index to use the optimized versions:
```bash
pip install --no-index torch
pip install --no-index matplotlib
pip install --no-index pybind11
pip install polyscope
pip install --no-index scipy scikit-image
pip install --no-index cython
pip install usd-core flask tornado comm ipyevents ipycanvas jupyter_client
pip install open3d
```

#### Local Machine

Install the pip packages listed above. The module command is not needed for local installations.

#### Compile and Install Other Dependencies

Create a virtual environment. Ensure all submodules are cloned:
```bash
git submodule update --init --recursive
```

Most dependencies can be installed via pip (see package list above). Note that some packages may not need compilation if you're using a standard Python interpreter version.

**PyTorch3D:**
```bash
cd 3rdparty/pytorch3d
python setup.py build
python setup.py install
```

**gDel3D:**
```bash
cd 3rdparty/gDel3D/python_bindings
mkdir build
cd build
cmake ..
make
cp pygdel3d.cpython-311-x86_64-linux-gnu.so <PATH_TO_VENV_SITE_PACKAGES>
```

**Kaolin:**
```bash
cd 3rdparty/kaolin
git apply ../kaolin.patch
python setup.py build
python setup.py install
```

**diffvoronoi module** is from Delaunay Parallel 3D:
https://github.com/BrunoLevy/geogram/wiki/Delaunay3D

#### Deprecated (or Broken)

**USD** -- can be installed with pip

**Open3D** -- can be installed with pip. Old instructions kept as backup:
```bash
cd 3rdparty/open3d
git apply ../open3d.patch
mkdir build
cd build
cmake -GNinja ..
cmake --build . --config Release --target install-pip-package
```

