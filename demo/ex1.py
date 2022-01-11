#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing the Mandelbrot Set Using Python by Blake Sanie, Nov 29, 2020.
https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

Use Python Numba for C-like performance, Jan 11, 2022.
Part 1 of 4 Just-In-Time (JIT) Compilation.
"""

from PIL import Image
from os.path import exists
from numba import njit, uint8
from timeit import default_timer as timer
import colorsys, os, sys
import numpy as np

# frame parameters
width = 700 # pixels
x, y = -0.5, 0.0
xRange = 3.4
aspectRatio = 4 / 3 
precision = 500

height = round(width / aspectRatio)
yRange = xRange / aspectRatio
minX = x - xRange / 2
maxX = x + xRange / 2
minY = y - yRange / 2
maxY = y + yRange / 2

jit_start = timer()

# JIT hsv_to_rgb (Note: return type UniTuple(f8,3) or None)
hsv_to_rgb = njit('(f8, f8, f8)', nogil=True)(colorsys.hsv_to_rgb)

@njit('UniTuple(u1,3)(f8, f8, f8, f8)', nogil=True)
def powerColor(distance, exp, const, scale):
    color = distance**exp
    r,g,b = hsv_to_rgb(const + scale * color, 1 - 0.6 * color, 0.9)
    r = uint8(r * 255 + 0.5)
    g = uint8(g * 255 + 0.5)
    b = uint8(b * 255 + 0.5)
    return r,g,b

@njit('void(u1[:,:,:])', nogil=True)
def mandel(pixels):
    for row in range(height):
        for col in range(width):
            x = minX + col * xRange / width
            y = maxY - row * yRange / height
            oldX = x
            oldY = y
            for i in range(precision + 1):
                a = x*x - y*y # real component of z^2
                b = 2 * x * y # imaginary component of z^2
                x = a + oldX  # real component of new z
                y = b + oldY  # imaginary component of new z
                if x*x + y*y > 4:
                    break
            if i < precision:
                distance = (i + 1) / (precision + 1)
                # Numpy ndarray is [row, col] versus Pillow Image [col, row].
                pixels[row, col] = powerColor(distance, 0.2, 0.27, 1.0)

def save_image(pixels, filename, show_image=False):
    height, width, dim = pixels.shape
    pixels = pixels.reshape((height * width * dim,))

    img = Image.frombuffer("RGB", (width, height), pixels, "raw", "RGB", 0, 1)
    img.save(filename)
    print(f"image saved as {filename}")

    if show_image:
        if sys.platform == "darwin":
            os.system(f"open {filename}")
        elif sys.platform.startswith("linux") and exists("/usr/bin/eog"):
            os.system(f"eog {filename}")

if __name__ == '__main__':
    print("     jit time {:.3f} seconds".format(timer() - jit_start))
    pixels = np.empty((height, width, 3), dtype=np.uint8)

    s = timer()
    mandel(pixels)
    print("   mandelbrot {:.3f} seconds".format(timer() - s))

    save_image(pixels, "img1.png", show_image=False)

