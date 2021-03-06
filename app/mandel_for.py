# -*- coding: utf-8 -*-
"""
Mandelbrot functions.
"""

import numpy as np

from .mandel_common import get_color, check_colors, mandel1, mandel2
from numba import njit, float32, uint8, int16

@njit('void(u1[:,:,:], i2[:,:], UniTuple(i4,2), f8, f8, f8, f8, i4)', nogil=True)
def mandelbrot1(temp, colors, seq, min_x, min_y, step_x, step_y, max_iters):

    width = temp.shape[1]

    for y in range(seq[0], seq[1]):
        cimag = min_y + (y * step_y)

        for x in range(width):
            creal = min_x + (x * step_x)
            temp[y,x] = mandel1(colors, creal, cimag, max_iters)


@njit('void(u1[:,:,:], i2[:,:], UniTuple(i4,2), f8, f8, f8, f8, i4, i4, f8[:], u1[:,:,:])', nogil=True)
def mandelbrot2(temp, colors, seq, min_x, min_y, step_x, step_y, max_iters, aafactor, offset, output):

    height, width = temp.shape[:2]
    aaarea = aafactor * aafactor
    aaarea2 = (aaarea - 1) * 2

    for y in range(seq[0], seq[1]):
        c = np.empty((3,), dtype=np.int32)

        for x in range(width):
            c1 = temp[y,x]
            count = False

            # Skip AA for colors within tolerance.
            if not count and x > 0:
                count = check_colors(c1, temp[y, x - 1])
            if not count and x + 1 < width:
                count = check_colors(c1, temp[y, x + 1])
            if not count and x > 1:
                count = check_colors(c1, temp[y, x - 2])
            if not count and x + 2 < width:
                count = check_colors(c1, temp[y, x + 2])

            if not count and y > 0:
                count = check_colors(c1, temp[y - 1, x])
            if not count and y + 1 < height:
                count = check_colors(c1, temp[y + 1, x])
            if not count and y > 1:
                count = check_colors(c1, temp[y - 2, x])
            if not count and y + 2 < height:
                count = check_colors(c1, temp[y + 2, x])

            if not count:
                output[y,x] = c1
                continue

            # Compute AA.
            c[:] = c1

            for i in range(0, aaarea2, 2):
                creal = min_x + ((x + offset[i]) * step_x)
                cimag = min_y + ((y + offset[i+1]) * step_y)
                color = mandel2(colors, creal, cimag, max_iters)
                c[0] += color[0]
                c[1] += color[1]
                c[2] += color[2]

            output[y,x] = (uint8(c[0]/aaarea), uint8(c[1]/aaarea), uint8(c[2]/aaarea))


@njit('void(f4[:], u1[:,:,:], u1[:,:,:], UniTuple(i4,2))', nogil=True)
def horizontal_gaussian_blur(matrix, src, dst, seq):

    height, width = src.shape[:2]
    cols = matrix.shape[0] >> 1

    for y in range(seq[0], seq[1]):
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


@njit('void(u1[:,:,:], u1[:,:,:], UniTuple(i4,2))', nogil=True)
def unsharp_mask(src, dst, seq):
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

    for y in range(seq[0], seq[1]):

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

