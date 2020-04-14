Installations instructions for SiMa.ai, on Mac.
===============================================

The original instructions to install TVM are here: https://docs.tvm.ai/install/from_source.html

Here, we will simplify the steps so to get you started quickly on your Macs.

# Pre-requisites
Install Homebrew, miniconda, create a tvm environment.

You can use pyenv too, just make sure to create a tvm environment.

```shell script
# install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew update; brew upgrade

# Install miniconda
brew cask install miniconda

# open a new terminal with Command-N

# create tvm environment
conda create -n tvm
conda activate tvm
```

Since Macs don't have Clang/LLVM with OpenMP, we install it
```shell script
conda install -y -c conda-forge llvmdev llvm-openmp llvm-tools clang clang-tools clang-dev
```

Install ANTLR parser for TensorFlow parsing
```shell script
conda install -y -c conda-forge antlr antlr-python-runtime
```

Install build tools (can also be done with brew for system-wide)
```shell script
conda install -y ninja cmake 
```

Install misc packages
```shell script
conda install -y -c conda-forge numpy decorator attrs tornado xgboost psutil cython
```
Not required by useful
```shell script
conda install -y -c conda-forge matplotlib opencv py-opencv yaml yapf python.app h5py hdf5 cffi
```

# Setup
```shell script
git clone --recursive https://github.com/apache/incubator-tvm tvm
cd tvm
git submodule init
git submodule update
```
Now you are in ```$TVM_HOME```

# Build
```shell script
cd $TVM_HOME
mkdir build
cd build
cp ../sima-config.cmake config.cmake
cmake ..
ninja
cd ..
make cython
```
It takes about 10mn to build and 2s for cython.

# Add TVM python paths to system-wide paths
```shell script
# at the end of ~/.zshrc, add
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
```

# PyCharm
- Point PyCharm to use your conda tvm environment.
- Add those 2 paths ```$TVM_HOME/python:$TVM_HOME/topi/python``` to your project.

