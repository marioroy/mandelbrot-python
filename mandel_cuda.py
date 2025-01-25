#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the GPU using PyCUDA.
"""

import io, os, sys
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH, RADIUS
from app.interface import WindowPygame

if sys.platform != 'win32':
    cpath = os.getenv('CPATH') or ''
    os.environ['CPATH'] = cpath + ':/usr/local/cuda/include'
    lpath = os.getenv('LIBRARY_PATH') or ''
    os.environ['LIBRARY_PATH'] = lpath + ':/usr/local/cuda/lib64:/usr/local/cuda/lib'

filepath = os.path.join(os.path.dirname(__file__), \
    'app', 'mandel_cuda.h').replace(' ', '\\ ')
with io.open(filepath, 'r', encoding='utf-8') as file:
    KERNEL_SOURCE = file.read()

if OPT.mixed_prec < 3:
    filepath = os.path.join(os.path.dirname(__file__), \
        'app', 'mandel_cuda_mp12.c').replace(' ', '\\ ')
    with io.open(filepath, 'r', encoding='utf-8') as file:
        KERNEL_SOURCE = KERNEL_SOURCE + file.read()
else:
    filepath = os.path.join(os.path.dirname(__file__), \
        'app', 'dfloat.h').replace(' ', '\\ ')
    with io.open(filepath, 'r', encoding='utf-8') as file:
        KERNEL_SOURCE = KERNEL_SOURCE + file.read()
    filepath = os.path.join(os.path.dirname(__file__), \
        'app', 'mandel_cuda_mp34.c').replace(' ', '\\ ')
    with io.open(filepath, 'r', encoding='utf-8') as file:
        KERNEL_SOURCE = KERNEL_SOURCE + file.read()

try:
    import pycuda.driver as cuda
except ModuleNotFoundError:
    print("[ERROR] No module named pycuda")
    sys.exit(1)

from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

class App(WindowPygame):

    def __init__(self, opt):
        super().__init__(opt)

        cuda.init()
        self.cuda_ctx = make_default_context()
        print("[GPU]", self.cuda_ctx.get_device().name())

        # Construct memory objects.
        self.colors = np.empty((GRADIENT_LENGTH, 3), dtype=np.ctypeslib.ctypes.c_int16)
        self.init_colors()

        self.offset = np.empty(((9*9-1)*2,), dtype=np.ctypeslib.ctypes.c_double)
        self.init_offset()

        h, w = self.height, self.width
        self.output = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)
        self.temp = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)
        self.temp2 = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)

        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        # Allocate CUDA variables. Here, I set -fmad to false.
        # Instead, FMA is determined by the FMA_ON or FMA_OFF switch.
        options = [
            '-allow-unsupported-compiler',
            '-fmad=true' if self.fma else '-fmad=false',
            '-prec-div=true', '-prec-sqrt=true',
            '--compiler-options',
            '-DFMA_ON' if self.fma else '-DFMA_OFF',
            '-DRADIUS={}'.format(RADIUS),
            '-DGRADIENT_LENGTH={}'.format(GRADIENT_LENGTH),
            '-DMATRIX_LENGTH={}'.format(self.gaussian_kernel.shape[0]),
            '-DMIXED_PREC{}'.format(self.mixed_prec) ]

        if len(OPT.compiler_bindir) > 0:
            # Use given bin path or compiler name.
            options.insert(0, '--compiler-bindir={}'.format(OPT.compiler_bindir))
        elif "NVCC_PREPEND_FLAGS" in os.environ:
            # Use system-wide or user setting e.g. -ccbin=/opt/cuda/bin
            pass
        elif os.path.exists('/usr/local/cuda/bin/gcc'):
            # Compile using GCC symbolic link, inside the cuda bin dir.
            options.insert(0, '--compiler-bindir=/usr/local/cuda/bin/gcc')
        elif os.path.exists('/usr/local/bin/gcc-13') or os.path.exists('/usr/bin/gcc-13'):
            # Compile using GCC 13, if available on the system.
            options.insert(0, '--compiler-bindir=gcc-13')
        elif os.path.exists('/usr/local/bin/gcc-12') or os.path.exists('/usr/bin/gcc-12'):
            # Compile using GCC 12, if available on the system.
            options.insert(0, '--compiler-bindir=gcc-12')
        elif os.path.exists('/usr/local/bin/gcc-11') or os.path.exists('/usr/bin/gcc-11'):
            # Compile using GCC 11, if available on the system.
            options.insert(0, '--compiler-bindir=gcc-11')
        elif os.path.exists('/usr/local/bin/gcc-10') or os.path.exists('/usr/bin/gcc-10'):
            # Compile using GCC 10, if available on the system.
            options.insert(0, '--compiler-bindir=gcc-10')

        try:
            self.cuda_prg = SourceModule(KERNEL_SOURCE, options=options)
        except Exception as e:
            print("[ERROR] pycuda:", e)
            self.cuda_ctx.pop()
            sys.exit(1)

        self.d_temp = cuda.mem_alloc(self.temp.nbytes)
        self.d_temp2 = cuda.mem_alloc(self.temp2.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)
        self.d_offset = cuda.mem_alloc(self.offset.nbytes)
        self.d_matrix = cuda.mem_alloc(self.gaussian_kernel.nbytes)
        self.d_colors = cuda.mem_alloc(self.colors.nbytes)
        cuda.memcpy_htod(self.d_matrix, self.gaussian_kernel)

        # Instantiate the Window interface.
        super().init()

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )

        bDimX = 8
        bDimY = 4
        gDimX = self.divide_up(self.width, bDimX)
        gDimY = self.divide_up(self.height, bDimY)

        if self.update_flag:
            cuda.memcpy_htod(self.d_offset, self.offset)
            cuda.memcpy_htod(self.d_colors, self.colors)
            self.update_flag = False

        # State 1.
        cuda_func = self.cuda_prg.get_function("mandelbrot1")
        cuda_func(
            np.float64(self.min_x), np.float64(self.min_y),
            np.float64(step_x), np.float64(step_y), self.d_temp, self.d_colors,
            np.int32(iters), np.int32(self.width), np.int32(self.height),
            block=(bDimX,bDimY,1), grid=(gDimX,gDimY), shared=0 )

        cuda.Context.synchronize()

        if self.num_samples == 1:
            cuda.memcpy_dtoh(self.temp, self.d_temp)
            self.update_window()
            return

        # State 2.
        cuda_func = self.cuda_prg.get_function("mandelbrot2")
        cuda_func(
            np.float64(self.min_x), np.float64(self.min_y),
            np.float64(step_x), np.float64(step_y), self.d_output, self.d_temp,
            self.d_colors, np.int32(iters), np.int32(self.width),
            np.int32(self.height), np.int16(self.num_samples), self.d_offset,
            block=(bDimX,bDimY,1), grid=(gDimX,gDimY), shared=0 )

        cuda.Context.synchronize()

        # Image sharpening.
        cuda.memcpy_dtod(self.d_temp, self.d_output, self.output.nbytes)

        bDimX = 24
        gDimX = self.divide_up(self.width, bDimX)

        cuda_func = self.cuda_prg.get_function("horizontal_gaussian_blur")
        cuda_func(
            self.d_matrix, self.d_output, self.d_temp2,
            np.int32(self.width), np.int32(self.height),
            block=(bDimX,bDimY,1), grid=(gDimX,gDimY), shared=0 )

        cuda.Context.synchronize()

        cuda_func = self.cuda_prg.get_function("vertical_gaussian_blur")
        cuda_func(
            self.d_matrix, self.d_temp2, self.d_output,
            np.int32(self.width), np.int32(self.height),
            block=(bDimX,bDimY,1), grid=(gDimX,gDimY), shared=0 )

        cuda.Context.synchronize()

        cuda_func = self.cuda_prg.get_function("unsharp_mask")
        cuda_func(
            self.d_temp, self.d_output,
            np.int32(self.width), np.int32(self.height),
            block=(bDimX,bDimY,1), grid=(gDimX,gDimY), shared=0 )

        cuda.Context.synchronize()
        cuda.memcpy_dtoh(self.output, self.d_output)
        self.update_window()

    def exit(self):

        self.cuda_ctx.pop()

        del self.d_colors, self.d_matrix, self.d_offset
        del self.d_output, self.d_temp, self.d_temp2
        del self.cuda_prg, self.cuda_ctx

        del self.colors, self.offset, self.output
        del self.temp, self.temp2


if __name__ == '__main__':

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

