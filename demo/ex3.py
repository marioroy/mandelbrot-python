#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing the Mandelbrot Set Using Python by Blake Sanie, Nov 29, 2020.
https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

Use Python Numba for C-like performance, Jan 11, 2022.
Part 3 of 4 Gaussian Blur 1-D, Two Dimensions.
"""

from numba import njit, float32, int16, uint8
from timeit import default_timer as timer
import numpy as np
import math, os, sys

jit_start = timer()

sys.path.insert(0, os.path.dirname(__file__))
from ex1 import width, height, save_image
from ex2 import mandelbrot1, mandelbrot2

def create_gaussian_blur_1d_kernel(sigma):
    """
    Returns 1D convolution kernel for Gaussian blur in two dimensions.
    https://en.wikipedia.org/wiki/Gaussian_blur
    """
    # minimum standard deviation
    if sigma < 0.2:
        sigma = 0.2

    radius = int(math.ceil(sigma)) * 2
    sigma22 = 2.0 * sigma * sigma
    sqrt_sigma_pi22 = math.sqrt(math.pi * sigma22)

    # increment radius by 1 if not close to zero
    distance = radius * radius
    if math.exp(-(distance) / sigma22) / sqrt_sigma_pi22 > 0.01:
        radius += 1

    size = radius * 2 + 1
    kernel = np.empty((size,), dtype=np.float32)
    radius2 = radius * radius
    i = 0

    # fill kernel
    for x in range(-radius, radius + 1):
        distance = x * x
        kernel[i] = math.exp(-(distance) / sigma22) / sqrt_sigma_pi22
        i += 1

    # filter
    kernel = kernel[np.where(kernel > 1e-17)]
    total = np.sum(kernel)

    # normalize
    for i in range(kernel.shape[0]):
        kernel[i] /= total

    return kernel

@njit('void(f4[:], u1[:,:,:], u1[:,:,:])', nogil=True)
def horizontal_gaussian_blur(matrix, src, dst):

    height, width = src.shape[:2]
    cols = matrix.shape[0] >> 1

    for y in range(height):
        for x in range(width):

          # r = g = b = float32(0.5)
          # for col in range(-cols, cols + 1):
          #     ix = x + col
          #     if ix < 0:
          #         ix = 0
          #     elif ix >= width:
          #         ix = width - 1
          #     rgb = src[y,ix]
          #     wgt = matrix[cols + col]
          #     r += float32(wgt * rgb[0])
          #     g += float32(wgt * rgb[1])
          #     b += float32(wgt * rgb[2])

            # Gaussian blur optimized 1-D loop.
            rgb = src[y,x]
            wgt = matrix[cols]
            r = float32(wgt * rgb[0] + 0.5)
            g = float32(wgt * rgb[1] + 0.5)
            b = float32(wgt * rgb[2] + 0.5)
            col2 = cols + cols

            for col in range(-cols, 0):
                ix = x + col
                if ix < 0:
                    ix = 0
                rgb = src[y,ix]

                ix = x + col + col2
                if ix >= width:
                    ix = width - 1
                rgb2 = src[y,ix]

                wgt = matrix[cols + col]
                r += float32(wgt * (int16(rgb[0]) + rgb2[0]))
                g += float32(wgt * (int16(rgb[1]) + rgb2[1]))
                b += float32(wgt * (int16(rgb[2]) + rgb2[2]))
                col2 -= 2

            dst[y,x] = (uint8(r), uint8(g), uint8(b))

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
    print(" antialiasing {:.3f} seconds".format(timer() - s))

    s = timer()
    horizontal_gaussian_blur(gkernel, pixels2, pixels3)
    horizontal_gaussian_blur(gkernel, pixels3_T, pixels2_T)
    print("gaussian blur {:.3f} seconds".format(timer() - s))

    save_image(pixels2, "img3.png", show_image=False)

