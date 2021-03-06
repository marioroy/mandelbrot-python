#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the CPU using Numba's parfor loop.

Until set_parallel_chunksize() is added to Numba, this script may run
slower than mandel_queue.py and mandel_stream.py. On the other hand,
this script is ready for set_parallel_chunksize() when made available.
See app/mandel_parfor.py.

Results taken from an 1920x1080 auto-zoom session on a 32-core box.
  Numba 0.54.1     10.6 seconds  slowest of the bunch
  Numba autochunk   8.9 seconds  fastest of the bunch
  mandel_ocl        9.5 seconds  OpenCL on the CPU
  mandel_queue      9.5 seconds  SimpleQueue
  mandel_stream     9.1 seconds  Socket
"""

import os
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH
from app.interface import WindowPygame

# Silently limit the number of threads to os.cpu_count().
NUM_THREADS = min(os.cpu_count(), max(1, OPT.num_threads))
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
        self.chunksize_h = self.divide_up(h, blur_num_threads)
        self.chunksize_w = self.divide_up(w, blur_num_threads)

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

