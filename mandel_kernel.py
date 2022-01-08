#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the GPU using cuda.jit.
"""

import math
import os
import sys
import numpy as np

from app.option import OPT

if sys.platform != 'win32':
    cpath = os.getenv('CPATH') or ''
    os.environ['CPATH'] = cpath + ':/usr/local/cuda/include'
    lpath = os.getenv('LIBRARY_PATH') or ''
    os.environ['LIBRARY_PATH'] = lpath + ':/usr/local/cuda/lib64:/usr/local/cuda/lib'

from numba import config, cuda
from numba.cuda.cudadrv import nvvm
from numba.core import utils

if config.ENABLE_CUDASIM:
    print("Script not supported in CUDA simulator, exiting...")
    sys.exit(1)
if utils.MACHINE_BITS == 32:
    print("CUDA not supported for 32-bit architecture, exiting...")
    sys.exit(1)
if not nvvm.is_available():
    print("CUDA libNVVM not available, exiting...")
    sys.exit(1)

from app.base import GRADIENT_LENGTH
from app.interface import WindowPygame

from app.mandel_kernel import \
    mandelbrot1, mandelbrot2, horizontal_gaussian_blur, \
    vertical_gaussian_blur, unsharp_mask

class App(WindowPygame):

    def __init__(self, opt):
        super().__init__(opt)

        print("[GPU]", cuda.get_current_device().name.decode("utf-8"))
        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        # Construct memory objects.
        self.colors = np.empty((GRADIENT_LENGTH, 3), dtype=np.ctypeslib.ctypes.c_int16)
        self.init_colors()

        self.offset = np.empty(((9*9-1)*2,), dtype=np.ctypeslib.ctypes.c_double)
        self.init_offset()

        h, w = self.height, self.width
        self.output = np.empty((h,w,3), dtype=np.ctypeslib.ctypes.c_uint8)
        self.temp = np.empty((h,w,3), dtype=np.ctypeslib.ctypes.c_uint8)

        # Allocate CUDA variables.
        stream = cuda.stream()
        with stream.auto_synchronize():
            self.d_temp = cuda.device_array((h,w,3), dtype=np.uint8, stream=stream)
            self.d_temp2 = cuda.device_array((h,w,3), dtype=np.uint8, stream=stream)
            self.d_output = cuda.device_array((h,w,3), dtype=np.uint8, stream=stream)
            self.d_offset = cuda.to_device(self.offset, stream=stream)
            self.d_matrix = cuda.to_device(self.gaussian_kernel, stream=stream)
            self.d_colors = cuda.to_device(self.colors, stream=stream)

        self.stream = stream

        # Instantiate the Window interface.
        super().init()

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )

        bDimX, bDimY = 8, 4
        gDimX = self.divide_up(self.width, bDimX)
        gDimY = self.divide_up(self.height, bDimY)

        stream = self.stream

        if self.update_flag:
            with stream.auto_synchronize():
                self.d_offset.copy_to_device(self.offset, stream=stream)
                self.d_colors.copy_to_device(self.colors, stream=stream)
            self.update_flag = False

        # State 1.
        with stream.auto_synchronize():
            mandelbrot1[(gDimX,gDimY),(bDimX,bDimY,1),stream](
                self.d_temp, self.d_colors, self.width, self.height,
                self.min_x, self.min_y, step_x, step_y, iters )

            if self.num_samples == 1:
                self.d_temp.copy_to_host(self.temp, stream=stream)
                self.update_window()
                return

        # State 2.
        with stream.auto_synchronize():
            mandelbrot2[(gDimX,gDimY),(bDimX,bDimY,1),stream](
                self.d_temp, self.d_colors, self.width, self.height,
                self.min_x, self.min_y, step_x, step_y, iters,
                self.num_samples, self.d_offset, self.d_output )

            # Image sharpening.
            self.d_temp.copy_to_device(self.d_output, stream=stream)

            bDimX = 24
            gDimX = self.divide_up(self.width, bDimX)

            horizontal_gaussian_blur[(gDimX,gDimY),(bDimX,bDimY,1),stream](
                self.d_matrix, self.d_output, self.d_temp2, self.width, self.height)

            vertical_gaussian_blur[(gDimX,gDimY),(bDimX,bDimY,1),stream](
                self.d_matrix, self.d_temp2, self.d_output, self.width, self.height)

            unsharp_mask[(gDimX,gDimY),(bDimX,bDimY,1),stream](
                self.d_temp, self.d_output, self.width, self.height)

            self.d_output.copy_to_host(self.output, stream=stream)
            self.update_window()

    def exit(self):

        del self.d_colors, self.d_offset, self.d_output
        del self.d_matrix, self.d_temp, self.d_temp2
        del self.stream
        cuda.close()

        del self.colors, self.offset, self.output, self.temp


if __name__ == '__main__':

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

