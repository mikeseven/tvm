Installations instructions for SiMa.ai, on Mac.
===============================================

The original instructions to install TVM are here: https://docs.tvm.ai/install/from_source.html

Here, we will simplify the steps so to get you started quickly on your Macs.

# Pre-requisites
## Prepare for conda environment
Install `Homebrew`, `miniconda`, and create a tvm environment under conda.

You can use `pyenv` too, just make sure to create a tvm environment.

```shell
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew update && brew upgrade

# Install miniconda
brew cask install miniconda

# Source conda.sh whenever opening a new terminal(Path to conda.sh may differ)
echo ". /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" >> ~/.bash_profile

# Open a new terminal with Command-N or make sure you source ~/.bash_profile

# Create tvm environment, make sure we use python3.7 (for now)
conda create -n tvm python=3.7
conda activate tvm
```

Now we are in tvm environment!

## Install required libraries
Since Macs don't have Clang/LLVM with OpenMP, we install it
```shell
conda install -y -c conda-forge llvmdev llvm-openmp llvm-tools clang clang-tools clangdev
```

Install ANTLR parser for TensorFlow parsing
```shell
conda install -y -c conda-forge antlr antlr-python-runtime
```

Install build tools (can also be done with brew for system-wide)
```shell
conda install -y ninja cmake 
```

Install misc packages
```shell
conda install -y -c conda-forge numpy decorator attrs tornado xgboost psutil cython
```

Install glfw3
```shell
conda install -y -c menpo glfw3
```

Not required by useful
```shell
conda install -y -c conda-forge matplotlib opencv py-opencv yaml yapf python.app h5py hdf5 cffi
```

Install tensorflow and tflite
```shell
# Make sure the pip command is pointing to conda
which pip
pip install tensorflow==1.14.0 tflite==2.1.0
```

# Setup
```shell
# Get into sima's TVM and update the repo
cd tvm
git submodule init
git submodule update
```
Now you are in ```$TVM_HOME```

# Build
```shell
cd $TVM_HOME
mkdir build
cd build
cp ../sima-config.cmake config.cmake
cmake -G Ninja ..
ninja
cd ..
make cython
```
It takes about 10mn to build and 2s for cython.

# Add TVM python paths to system-wide paths
```shell
# at the end of ~/.zshrc, add
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
```

# PyCharm
- Point PyCharm to use your conda tvm environment.
- Add those 2 paths ```$TVM_HOME/python:$TVM_HOME/topi/python``` to your project.