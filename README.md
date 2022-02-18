# Mandelbrot Set Explorer

*This is no longer actively maintained and kept here for reference.*

A demonstration for exploring the [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set)
using Python. Similarly, CUDA and OpenCL solutions for comparison.

<p align="center">
  <img src="../assets/mandelbrot.png?raw=true" alt="Mandelbrot Set"/>
</p>

**Requirements and Installation**

This requires Python 3.6+ minimally, Numba, Numpy, and Pygame.
The Numba installion pulls Numpy specific to the Numba release.
Optionally, install Pyopencl and/or Pycuda for running on the GPU.

Development was done in a Linux environment. However, you will
be pleased to know that testing was done on FreeBSD, Linux, macOS,
and Microsoft Windows.

```bash
# optional packages if building numba from source
$ pip3 install icc_rt intel_openmp tbb tbb4py
```

```bash
$ pip3 install numba pygame
$ pip3 install pyopencl  # optional, for CPU/GPU
$ pip3 install pycuda    # optional, for GPU
```

**Demo Folder**

Non-parallel demonstrations for creating the Mandelbrot Set, apply Anti-Aliasing,
Gaussian Blur, and Unsharp Mask via a step-by-step approach.

**Python Scripts**

Rendering on the GPU requires double-precision support on the device.
On FreeBSD, install `pocl` for OpenCL on the CPU. It works quite well.
On Windows, run `vcvars64.bat` first (Visual Studio) for CUDA/OpenCL.

```text
mandel_queue.py   - Run parallel using a queue for IPC
mandel_stream.py  - Run parallel using a socket for IPC
mandel_parfor.py  - Run parallel using Numba's parfor loop
mandel_ocl.py     - Run on the CPU or GPU using PyOpenCL
mandel_cuda.py    - Run on the GPU using PyCUDA (GCC 10.x max)
mandel_kernel.py  - Run on the GPU using cuda.jit

$ python3 mandel_queue.py -h
$ python3 mandel_queue.py --shortcuts
$ python3 mandel_queue.py --config=app.ini 720p
$ python3 mandel_queue.py --config=app.ini 720p --num-samples=3
$ python3 mandel_queue.py --location 5
```

Until CUDA reaches full compatibility with GCC 11.x, optionally specify
GCC 10.x or lower for the `mandel_cuda.py` demonstration. On Clear Linux,
install the `c-extras-gcc10` bundle.

```text
$ sudo swupd bundle-add c-extras-gcc10

$ python3 mandel_cuda.py --compiler-bindir=/usr/bin/gcc-10
$ python3 mandel_cuda.py --compiler-bindir=gcc-10
```

**Auto-Zoom Destinations**

<p align="center">
  <img src="../assets/locations.png?raw=true" alt="Auto-Zoom Locations"/>
</p>

**Color Schemes**

<p align="center">
  <img src="../assets/colorschemes.png?raw=true" alt="Color Schemes"/>
</p>

# Usage

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
    --mixed-prec=ARG   select mixed-precision flag [0,1,2]: 0
    --fma=ARG          select fused-multiply-add flag [0,1]: 0
```

Values exceeding the range specification are silently clipped to
the respective minimum or maximum value. The number of iterations
is computed dynamically based on the performance level
(lower equals more iterations).

# Keyboard Shortcuts

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

# Acknowledgements

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

