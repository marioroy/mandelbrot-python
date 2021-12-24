#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the CPU using Numba's parfor loop.
"""

import math
import os
import sys
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH
from app.interface import WindowPygame

# Silently limit the number of threads to os.cpu_count() - 1.
NUM_THREADS = min(os.cpu_count() - 1, max(1, OPT.num_threads))
os.environ['NUMBA_NUM_THREADS'] = str(NUM_THREADS)

from app.mandel_parfor import \
    mandelbrot1, mandelbrot2, horizontal_gaussian_blur, unsharp_mask

class App(WindowPygame):

    def __init__(self, opt):
        super().__init__(opt)

        print("[CPU] number of threads {}".format(NUM_THREADS))
        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        # Construct memory objects.
        self.colors = np.empty((GRADIENT_LENGTH, 3), dtype=np.ctypeslib.ctypes.c_int16)
        self.init_colors()

        self.offset = np.empty(((9*9-1)*2,), dtype=np.ctypeslib.ctypes.c_double)
        self.init_offset()

        h, w = self.height, self.width
        self.output = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)
        self.temp = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)
        self.temp2 = np.empty((h, w, 3), dtype=np.ctypeslib.ctypes.c_uint8)

        # Set chunksize for the CPU.
        blur_num_threads = max(1, min(NUM_THREADS, self.divide_up(h, 20)))
        blur_num_threads = max(1, min(blur_num_threads, os.cpu_count() // 2))
        self.chunksize_h = self.divide_up(self.height, blur_num_threads)
        self.chunksize_w = self.divide_up(self.width, blur_num_threads)

        # Instantiate the Window interface.
        super().init()

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )

        kernel = self.gaussian_kernel
        seqH = (0, self.height, self.chunksize_h)
        seqW = (0, self.width, self.chunksize_w)

        for state in range(1, 2 if self.num_samples == 1 else 3):
            if state == 1:
                mandelbrot1(
                    self.temp, self.colors, seqH, self.min_x, self.min_y,
                    step_x, step_y, iters )
            else:
                mandelbrot2(
                    self.temp, self.colors, seqH, self.min_x, self.min_y,
                    step_x, step_y, iters, self.num_samples, self.offset,
                    self.output )

                self.temp[:] = self.output[:]
                output_T = self.output.swapaxes(0,1)  # transpose(1,0,2)
                temp2_T = self.temp2.swapaxes(0,1)

                horizontal_gaussian_blur(kernel, self.output, self.temp2, seqH)
                horizontal_gaussian_blur(kernel, temp2_T, output_T, seqW)
                unsharp_mask(self.temp, self.output, seqH)

        self.update_window()

    def exit(self):

        del self.colors, self.offset, self.output
        del self.temp, self.temp2


if __name__ == '__main__':

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

