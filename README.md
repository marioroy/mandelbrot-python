# Mandelbrot Set Explorer

A demonstration for exploring the [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set) using Python. Similarly, CUDA and OpenCL solutions for comparison.

<p align="center">
  <img src="../assets/mandelbrot.png?raw=true" alt="Mandelbrot Set"/>
</p>

## Requirements and Installation

This requires Python 3.7 minimally, Numba, Numpy, and Pygame. Install Pyopencl or Pycuda for running on the GPU.

**Clear Linux**

The `c-basic` bundle provides the minimum development components.

```bash
sudo swupd bundle-add c-basic wget
```

For NVIDIA graphics, refer to [nvidia-driver-on-clear-linux](https://github.com/marioroy/nvidia-driver-on-clear-linux) for installing the driver and CUDA Toolkit.

**Ubuntu Linux 20.04.x**

Install the `build-essential` package for the build components. Optionally, install `pocl-opencl-icd` for running OpenCL on the CPU.

```bash
sudo apt update
sudo apt install build-essential
sudo apt install clinfo ocl-icd-libopencl1 ocl-icd-opencl-dev
sudo apt install opencl-c-headers opencl-clhpp-headers opencl-headers

sudo apt install pocl-opencl-icd
```

Using NVIDIA graphics? Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) for the `pycuda` demonstration. Important: Choose the CUDA Toolkit matching your display driver. I happen to be running the 515.x.x driver, so selected CUDA Toolkit 11.7.1. Your version may differ from mine. Adjust the paths accordingly.

| Driver | CUDA Toolkit |
|--------|--------------|
|  530   |    12.1.0    |
|  525   |    12.0.1    |
|  520   |    11.8.0    |
|  515   |    11.7.1    |
|  510   |    11.6.2    |

```bash
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run

# install CUDA Toolkit
sudo sh cuda_11.7.1_515.65.01_linux.run \
  --toolkit --installpath=/opt/cuda-11.7.1 \
  --no-opengl-libs --no-drm --override --silent

# remove OpenCL libs as they conflict with ocl-icd-libopencl1 package
sudo rm -f /opt/cuda-11.7.1/targets/x86_64-linux/lib/libOpenCL.so*

# create symbolic link
sudo ln -sf /opt/cuda-11.7.1 /opt/cuda

# update dynamic linker cache
sudo ldconfig

# update ~/.profile so that it can find nvcc
export PATH=$PATH:/opt/cuda/bin

# log out and log in; check that nvcc is in your path
which nvcc
```

## Miniconda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
for your platform using the default options.

On Windows, install [Build Tools for Visual Studio 2019](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers). Select only "Desktop Environment with C++" when installing. I tried version 16.11.25 from March 14, 2023, having no issues.

Open a shell with Miniconda activated. On Windows, launch "Anaconda Prompt (miniconda3)" from the Start Menu. Optionally, execute the `vcvars64.bat` command, including the quotes around it. It configures the VC build environment. The VC environment is required for installing `pycuda` and running the `pyopencl` and `pycuda` demonstrations.

```text
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

**Installation steps**

This involves `pip` for packages not available in the main channel.

```bash
conda install numpy==1.21.5   # not installed by default in miniconda

conda install tbb==2021.7.0   # not installed by default in miniconda
conda install tbb-devel==2021.7.0

conda install -c numba llvmlite numba
conda install -c numba llvmlite==0.39.1 numba==0.56.4  # or specific release
conda install -c numba/label/dev llvmlite numba        # or dev release

conda install pillow

pip install pygame
```

Install dependencies for `pyopencl` and `pycuda`. It requires OpenCL
and CUDA development files on the system to build successfully.
Ensure the `nvcc` command is in your path for `pycuda`.

```bash
conda install appdirs platformdirs MarkupSafe mako typing-extensions

pip install pytools   # another dependency
pip install pyopencl  # optional, for CPU or GPU
pip install pycuda    # optional, for NVIDIA GPU
```

The `pycuda` module may require manual installation on Unix platforms. Adjust the root path accordingly.

```bash
export CUDA_ROOT=/opt/cuda
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$PATH:$CUDA_ROOT/bin

cd ~/Downloads

# Obtain pycuda file from pypi.org
tar xf pycuda-2022.2.2.tar.gz
cd pycuda-2022.2.2

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
mandel_kernel.py  - Run on the GPU using cuda.jit

python3 mandel_queue.py -h
python3 mandel_queue.py --shortcuts
python3 mandel_queue.py --config=app.ini 720p
python3 mandel_queue.py --config=app.ini 720p --num-samples=3
python3 mandel_queue.py --location 5
```

The `mandel_cuda.py` example allows GCC 11 or higher, since Dec 28, 2022.
This is done by passing the `-allow-unsupported-compiler` option to `nvcc`.

Specify GCC 10.x or lower if unable to run due to GCC failure.

```text
# GCC 10 installation on Clear Linux
sudo swupd bundle-add c-extras-gcc10

python3 mandel_cuda.py --compiler-bindir=/usr/bin/gcc-10
python3 mandel_cuda.py --compiler-bindir=gcc-10
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
    --compiler-bindir  directory in which the C compiler resides
                 also, the compiler executable name can be specified

  GPU Options (mandel_cuda, mandel_ocl):
    --mixed-prec=ARG   select mixed-precision flag [0,1,2]: 2
    --fma=ARG          select fused-multiply-add flag [0,1]: 0
```

Values exceeding the range specification are silently clipped to
the respective minimum or maximum value. The number of iterations
is computed dynamically based on the performance level
(lower equals more iterations).

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
[) PageDn) zoom in from the center of the window
]) PageUp) zoom out from the center of the window
e)         export the window RGB values to image.png
a) Left)   scroll window left
s) Down)   scroll window down
d) Right)  scroll window right
w) Up)     scroll window up
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

**Color scheme**

```text
F1-F7)  select color scheme 1 through 7, respectively
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

