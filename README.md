# Mandelbrot Set Explorer

A demonstration for exploring the [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set) using Python. Similarly, CUDA and OpenCL solutions for comparison.

<p align="center">
  <img src="../assets/mandelbrot.png?raw=true" alt="Mandelbrot Set"/>
</p>

## Preparation

On GNU/Linux or UNIX derivative, refer to the OS specific instructions for CUDA installation. Depending on the CUDA version, GCC 13 or older is required and must be in your path before newer GCC releases.

On Windows, install [Build Tools for Visual Studio 2019](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers). Select only "Desktop Environment with C++" when installing. I tried version 16.11.25 from March 14, 2023, having no issues.

Open a shell with Miniforge activated. On Windows, launch "Anaconda Prompt (miniforge3)" from the Start Menu. Optionally, execute the `vcvars64.bat` command, including the quotes around it. It configures the VC build environment. The VC environment is required for installing `pycuda` and running the `pyopencl` and `pycuda` demonstrations.

```text
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

## Installation using Miniforge

First, install [Miniforge](https://conda-forge.org/download/) for your platform using the default options. Re-open your command shell to begin the installation. If you'd prefer that Conda's base environment not be activated on startup, run the following command when Conda is activated.

```bash
conda config --set auto_activate_base false
```

**Quick installation** using requirements file.

Note: Choose one Intel Math Kernel Library (MKL) or OpenBLAS support for NumPy.

```bash
# Create Conda environment.
conda env create -n mandel --file reqs_mkl.yml
conda env create -n mandel --file reqs_openblas.yml

# Activate the Conda environment.
conda activate mandel

# Prevent Conda from switching the BLAS library, choose same one.
conda config --env --add pinned_packages "blas=*=mkl"
conda config --env --add pinned_packages "blas=*=openblas"

# Pin the Numba version. mandel_kernel.py fails with 0.61.0.
conda config --env --add pinned_packages "numba==0.60.0"

# Continue to "Using vendor-supplied OpenCL drivers"
```

**Manual installation** allowing specific Python/Numba versions.

Note: Choose one MKL or OpenBLAS support for NumPy.

```bash
# Create a Conda environment.
conda create -n mandel python=3.12 "blas=*=mkl"
conda create -n mandel python=3.12 "blas=*=openblas"

# Activate the Conda environment.
conda activate mandel

# Prevent Conda from switching the BLAS library, choose same one.
conda config --env --add pinned_packages "blas=*=mkl"
conda config --env --add pinned_packages "blas=*=openblas"

# Pin the Numba version. mandel_kernel.py fails with 0.61.0.
conda config --env --add pinned_packages "numba==0.60.0"

# Install Numba.
conda install numba

# The CUDA target built-in to Numba is deprecated, with further
# development moved to the NVIDIA numba-cuda package.
conda install numba-cuda=0.3.0

# Install dependencies.
conda install appdirs platformdirs siphash24 tbb tbb-devel
conda install MarkupSafe mako pytools typing-extensions

# Install PyOpenCL from miniforge versus pip. This way, searching ICDs
# in <conda-root>/envs/<env-name>/etc/OpenCL/vendors/ is possible.
conda install pyopencl
```

The imaging library and library for drawing and handling keyboard events
have many dependencies. You can try `conda install pillow pygame` to
see the list. Optionally, answer `n` to exit the installation.

Alternately, install Pillow and Pygame via pip.

```bash
pip install pillow pygame
```

## Using vendor-supplied OpenCL drivers

Using AMD graphics? Install the ROCm OpenCL Runtime via the OS package manager.

```bash
# Arch Linux, CachyOS
sudo pacman -S rocm-opencl-runtime
```

Install a Conda Forge package for the OpenCL ICD loader to look
in `/etc/OpenCL/vendors` or copy the ICD files. Choose one.

```bash
# Install a package to include searching the `/etc` path
conda install ocl-icd-system

# Or copy the ICD file(s) to the Conda environment path
cp /etc/OpenCL/vendors/*.icd "$CONDA_PREFIX"/etc/OpenCL/vendors/
```

A benefit using Intel's OpenCL CPU runtime is auto-vectorization.

```bash
conda install intel-opencl-rt  # on x86-64 Linux or Windows
```

## PyCUDA Installation

Choose `conda` or `pip` for the PyCUDA installation. The former requires
the least amount of effort. Simply install `cuda-nvcc` matching or older
than your display driver. The latter, pip installation, requires `nvcc`
in your path and have a supported GCC version.

```bash
# Choose one only, conda or pip.
# Do not install cuda-nvcc newer than your display driver.

conda install pycuda cuda-nvcc==12.8.*   # running 570 driver 
conda install pycuda cuda-nvcc==12.6.*   # running 560 driver 
conda install pycuda cuda-nvcc==12.4.*   # running 550 driver 
conda install pycuda cuda-nvcc==12.2.*   # running 535 driver 

# Pin the CUDA version to prevent it from updating.
# Specify the same version used in the prior step.

conda config --env --add pinned_packages "cuda-nvcc==12.4.*"

# Alternatively, install pycuda via pip.

pip install pycuda   # prefer system CUDA Toolkit installation
```

In the event `pip` failed, you may try manual installation on Unix platforms.
Adjust the path accordingly. Do this in the same Conda environment.

```bash
export CUDA_ROOT=/opt/cuda
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$CUDA_ROOT/bin:$PATH

# Set an environment to supported GCC path
export NVCC_PREPEND_FLAGS="-ccbin=/path/to/supported/gcc/bin"

# Another way is making symbolic links to supported compilers
# and have the CUDA bin path searched first in your PATH env
sudo ln -sf /usr/bin/gcc-12 $CUDA_ROOT/bin/gcc
sudo ln -sf /usr/bin/g++-12 $CUDA_ROOT/bin/g++

cd ~/Downloads

# Obtain pycuda file from pypi.org
tar xf pycuda-2024.1.2.tar.gz
cd pycuda-2024.1.2

./configure.py
make install
```

## Python Scripts

Rendering on the GPU requires double-precision support on the device.
On FreeBSD, install `pocl` for OpenCL on the CPU. It works quite well.
On Windows, run the build tools `vcvars64.bat` first for CUDA/OpenCL.

```text
mandel_queue.py   - Run parallel using a queue for IPC
mandel_stream.py  - Run parallel using a socket for IPC
mandel_parfor.py  - Run parallel using Numba's parfor loop
mandel_ocl.py     - Run on the CPU or GPU using PyOpenCL
mandel_cuda.py    - Run on the GPU using PyCUDA
mandel_kernel.py  - Run on the GPU using Numba cuda.jit

python3 mandel_queue.py -h
python3 mandel_queue.py --shortcuts
python3 mandel_queue.py --config=app.ini 720p
python3 mandel_queue.py --config=app.ini 720p --num-samples=3
python3 mandel_queue.py --location 5
```

The `mandel_cuda.py` example may require an older GCC if using the
system CUDA installation.

```text
# Checks /usr/local/cuda/bin/gcc, gcc-13, gcc-12, gcc-11, gcc-10, or gcc.
# Typically, /usr/local/cuda is a symbolic link to /opt/cuda* path.

python3 mandel_cuda.py
python3 mandel_cuda.py --ccbin=/usr/bin/gcc-12
python3 mandel_cuda.py --ccbin=gcc-12
```

## Usage

Settings can be stored in a configuation file. If choosing to use a
configuration file, copy `app.ini` to `user.ini` and use that.
The `user.ini` file is ignored by git updates.

```text
Usage: mandel_queue.py [--config filepath [section]] [options]

Options:
  --version            show program's version number and exit
  -h, --help           show this help message and exit
  --shortcuts          show keyboard shortcuts and exit
  --width=ARG          width of window [100-8000]: 800
  --height=ARG         height of window [100-5000]: 500
  --location=ARG       begin at app location [0-7]: 0
  --center-x=ARG       home center-x value [float]: -0.625
  --center-y=ARG       home center-y value [float]: 0.0
  --zoom-scale=ARG     home zoom magnification [float]: 0.95
  --num-samples=ARG    anti-aliasing factor [1-9]: 2
  --perf-level=ARG     performance level [1-64]: 25
  --color-scheme=ARG   select color scheme [1-7]: 1
  --fast-zoom=ARG      select fast zoom [0,1]: 1
  --smooth-bench=ARG   select smooth bench [0,1]: 0

  CPU Options (mandel_parfor, mandel_queue, mandel_stream):
    --num-threads=ARG  number of threads to use: auto

  CUDA Options (mandel_cuda):
    --ccbin            directory in which the C compiler resides
                       (or) specify the compiler executable name

  GPU Options (mandel_cuda, mandel_ocl):
    --fma=ARG          select fused-multiply-add flag [0,1]: 0
    --mixed-prec=ARG   select mixed-precision flag [0,1,2,3,4]: 2

    Depending on the architecture, mixed-prec=1 may run faster than 2.
    0: double precision, fma=0 matching CPU results
    1,2: slight mixed-precision and still matching CPU results (fma=0)
    3: 1st sample double precision, supersampling float-float precision
    4: 1st sample and supersampling float-float precision
    mixed-prec=3,4 silently enables fma

Epilog:
  Values exceeding the range specification are silently clipped
  to the respective minimum or maximum value.

  The number of iterations is computed dynamically based on the
  performance level (lower equals more iterations).
```

**Auto-zoom destinations**

<p align="center">
  <img src="../assets/locations.png?raw=true" alt="Auto-Zoom Locations"/>
</p>

**Color schemes**

<p align="center">
  <img src="../assets/colorschemes.png?raw=true" alt="Color Schemes"/>
</p>

## Keyboard Shortcuts

Most all options are accessible via a keyboard shortcut.
Zooming does not exceed double-precision limit.

```text
q)         terminate the application and exit
r) Home)   reset window back to the home location
]) PageDn) zoom in from the center of the window
[) PageUp) zoom out from the center of the window
e)         export the window RGB values to image.png
a) Left)   scroll window left
s) Down)   scroll window down
d) Right)  scroll window right
w) Up)     scroll window up
```

**Color scheme**

```text
F1-F7)  select color scheme 1 through 7, respectively
```

**Auto zoom**

Specify option `--fast-zoom` for fast/slow zoom mode.
Specify option `--smooth-bench` for smooth/bench mode.
Press any key to end auto zoom.

```text
z)  auto zoom from scale 0.33 to near --location 1
x)  auto zoom from scale 0.33 to near --location 2
c)  auto zoom from scale 0.33 to near --location 3
v)  auto zoom from scale 0.33 to near --location 4
b)  auto zoom from scale 0.33 to near --location 5
g)  auto zoom from scale 0.33 to near --location 6
t)  auto zoom from scale 0.33 to near --location 7
```

**Go to location**

Display the image at location selection.

```text
shift-z) location 1
shift-x) location 2
shift-c) location 3
shift-v) location 4
shift-b) location 5
shift-g) location 6
shift-t) location 7
```

**Image quality**

Supersampling involves sharpening using Unsharp Mask.

```text
l)ow     render without supersampling
n)       render with 2x2 supersampling
m)edium  render with 3x3 supersampling
h)igh    render with 5x5 supersampling
u)ltra   render with 7x7 supersampling
i)       render with 9x9 supersampling
```

**Performance level**

Iterations are computed dynamically based on the performance level.

```text
<) ,)  decreasing performance level increases iterations ;min 1
>) .)  increasing performance level decreases iterations ;max 64
```

**Scroll movement**

Double clicking sets center-x,center-y to the mouse x,y coordinates
and renders the image.

```text
`)  set scroll factor to 0.002x for micro adjustments
1)  set scroll factor to 0.1x width or height
2)  set scroll factor to 0.2x width or height
3)  set scroll factor to 0.3x width or height
4)  set scroll factor to 0.4x width or height
5)  set scroll factor to 0.5x width or height
6)  set scroll factor to 0.6x width or height
7)  set scroll factor to 0.7x width or height
8)  set scroll factor to 0.8x width or height
9)  set scroll factor to 0.9x width or height
0)  set scroll factor to 1.0x width or height
```

## Demo Folder

The demo folder contains non-parallel demonstrations for creating the
Mandelbrot Set, apply Anti-Aliasing, Gaussian Blur, and Unsharp Mask
via a step-by-step approach.

## Acknowledgements

Bright qualitative colour scheme (courtesy of Paul Tol). I also tried
coloring using a slight modification of Bernstein polynomials.

* https://personal.sron.nl/~pault/

* https://mathworld.wolfram.com/BernsteinPolynomial.html

Periodicity checking i.e. escape early if we detect repetition.
Compute 2 more iterations to decrease the error term.

* http://locklessinc.com/articles/mandelbrot/

* http://linas.org/art-gallery/escape/escape.html

Unsharp Mask was adapted from Pillow code and made to run parallel.

* https://github.com/python-pillow/Pillow/blob/main/src/libImaging/UnsharpMask.c

