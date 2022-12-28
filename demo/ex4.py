#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing the Mandelbrot Set Using Python by Blake Sanie, Nov 29, 2020.
https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

Use Python Numba for C-like performance, Jan 11, 2022.
Part 4 of 4 Unsharp Mask.
"""

from numba import njit, float32, int16
from timeit import default_timer as timer
import numpy as np
import os, sys

jit_start = timer()

sys.path.insert(0, os.path.dirname(__file__))
from ex1 import width, height, save_image
from ex2 import mandelbrot1, mandelbrot2
from ex3 import create_gaussian_blur_1d_kernel, horizontal_gaussian_blur

@njit('void(u1[:,:,:], u1[:,:,:])', nogil=True)
def unsharp_mask(src, dst):
    """
    Sharpen the destination image using the Unsharp Mask technique.

    Compare "normal" pixel from "src" to "blurred" pixel from "dst".
    If the difference is more than threshold value, apply the OPPOSITE
    correction to the amount of blur, multiplied by percent.
    """
    height, width = src.shape[:2]
    percent = float32(65.0 / 100)
    threshold = int16(0)

    # Python version of C code plus multi-core CPU utilization.
    # https://github.com/python-pillow/Pillow/blob/main/src/libImaging/UnsharpMask.c

    for y in range(height):

        for x in range(width):
            norm_pixel = src[y,x]
            blur_pixel = dst[y,x]

            # Compare in/out pixels, apply sharpening.
            diff = int16(int16(norm_pixel[0]) - blur_pixel[0])
            if abs(diff) > threshold:
                # Add the difference to the original pixel.
                r = min(255, max(0, int16(diff * percent + norm_pixel[0])))
            else:
                # New pixel is the same as the original pixel.
                r = norm_pixel[0]

            diff = int16(int16(norm_pixel[1]) - blur_pixel[1])
            if abs(diff) > threshold:
                g = min(255, max(0, int16(diff * percent + norm_pixel[1])))
            else:
                g = norm_pixel[1]

            diff = int16(int16(norm_pixel[2]) - blur_pixel[2])
            if abs(diff) > threshold:
                b = min(255, max(0, int16(diff * percent + norm_pixel[2])))
            else:
                b = norm_pixel[2]

            dst[y,x] = (r,g,b)

if __name__ == '__main__':
    print("     jit time {:.3f} seconds".format(timer() - jit_start))
    pixels1 = np.empty((height, width, 3), dtype=np.uint8)
    pixels2 = np.empty((height, width, 3), dtype=np.uint8)
    pixels3 = np.empty((height, width, 3), dtype=np.uint8)

    gkernel = create_gaussian_blur_1d_kernel(2.0)
    pixels2_T = pixels2.swapaxes(0,1)  # transpose(1,0,2)
    pixels3_T = pixels3.swapaxes(0,1)

    s = timer()
    mandelbrot1(pixels1)
    print("   mandelbrot {:.3f} seconds".format(timer() - s))

    s = timer()
    mandelbrot2(pixels1, pixels2)
    print("anti-aliasing {:.3f} seconds".format(timer() - s))

    s = timer()
    pixels1[:] = pixels2[:] # make a copy for unsharp mask
    horizontal_gaussian_blur(gkernel, pixels2, pixels3)
    horizontal_gaussian_blur(gkernel, pixels3_T, pixels2_T)
    print("gaussian blur {:.3f} seconds".format(timer() - s))

    s = timer()
    unsharp_mask(pixels1, pixels2)
    print(" unsharp mask {:.3f} seconds".format(timer() - s))

    save_image(pixels2, "img4.png", show_image=False)

