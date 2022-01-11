#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing the Mandelbrot Set Using Python by Blake Sanie, Nov 29, 2020.
https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

Use Python Numba for C-like performance, Jan 11, 2022.
Part 2 of 4 Antialiasing.
"""

from numba import njit, int16
from timeit import default_timer as timer
import numpy as np
import os, sys

jit_start = timer()

sys.path.insert(0, os.path.dirname(__file__))
from ex1 import width, height, precision, minX, maxY, xRange, yRange
from ex1 import hsv_to_rgb, powerColor, save_image

@njit('b1(u1[:], u1[:])', nogil=True)
def check_colors(c1, c2):
    # return false if the colors are within tolerance
    if abs(int16(c2[0]) - c1[0]) > 8: return True
    if abs(int16(c2[1]) - c1[1]) > 8: return True
    if abs(int16(c2[2]) - c1[2]) > 8: return True
    return False

@njit('i4(f8, f8)', nogil=True)
def mandel(x, y):
    oldX = x
    oldY = y
    for i in range(precision + 1):
        a = x*x - y*y # real component of z^2
        b = 2 * x * y # imaginary component of z^2
        x = a + oldX  # real component of new z
        y = b + oldY  # imaginary component of new z
        if x*x + y*y > 4:
            break
    return i

@njit('void(u1[:,:,:])', nogil=True)
def mandelbrot1(pixels):
    for row in range(height):
        for col in range(width):
            x = minX + col * xRange / width
            y = maxY - row * yRange / height
            i = mandel(x, y)
            if i < precision:
                distance = (i + 1) / (precision + 1)
                pixels[row, col] = powerColor(distance, 0.2, 0.27, 1.0)

@njit('void(u1[:,:,:], u1[:,:,:])', nogil=True)
def mandelbrot2(pixels1, pixels2):
    aafactor = 7 # 7x7
    aareach1 = int(aafactor / 2.0)
    aareach2 = aareach1 + 1 if (aafactor % 2) else aareach1
    aaarea = int(aafactor * aafactor)
    aafactorinv = float(1.0 / aafactor)

    for row in range(height):
        c = np.empty((3,), dtype=np.int32)

        for col in range(width):
            c1 = pixels1[row, col]
            count = False

            # skip AA for colors within tolerance
            if not count and col > 0:
                count = check_colors(c1, pixels1[row, col - 1])
            if not count and col + 1 < width:
                count = check_colors(c1, pixels1[row, col + 1])
            if not count and col > 1:
                count = check_colors(c1, pixels1[row, col - 2])
            if not count and col + 2 < width:
                count = check_colors(c1, pixels1[row, col + 2])

            if not count and row > 0:
                count = check_colors(c1, pixels1[row - 1, col])
            if not count and row + 1 < height:
                count = check_colors(c1, pixels1[row + 1, col])
            if not count and row > 1:
                count = check_colors(c1, pixels1[row - 2, col])
            if not count and row + 2 < height:
                count = check_colors(c1, pixels1[row + 2, col])

            if not count:
                pixels2[row, col] = c1
                continue

            # compute AA
            c[:] = c1

            for yi in range(-aareach1, aareach2, 1):
                y = maxY - (row + yi*aafactorinv) * yRange / height

                for xi in range(-aareach1, aareach2, 1):
                    if (xi | yi) == 0: continue
                    x = minX + (col + xi*aafactorinv) * xRange / width

                    i = mandel(x, y)
                    if i < precision:
                        distance = (i + 1) / (precision + 1)
                        r, g, b = powerColor(distance, 0.2, 0.27, 1.0)
                        c[0] += r; c[1] += g; c[2] += b

            c2 = int(c[0]/aaarea), int(c[1]/aaarea), int(c[2]/aaarea)
            pixels2[row, col] = c2

if __name__ == '__main__':
    print("     jit time {:.3f} seconds".format(timer() - jit_start))
    pixels1 = np.empty((height, width, 3), dtype=np.uint8)
    pixels2 = np.empty((height, width, 3), dtype=np.uint8)

    s = timer()
    mandelbrot1(pixels1)
    print("   mandelbrot {:.3f} seconds".format(timer() - s))

    s = timer()
    mandelbrot2(pixels1, pixels2)
    print(" antialiasing {:.3f} seconds".format(timer() - s))

    save_image(pixels2, "img2.png", show_image=False)

