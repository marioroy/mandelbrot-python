#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing the Mandelbrot Set Using Python by Blake Sanie, Nov 29, 2020.
https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

Base code (no-JIT); Adapted from Blake Sanie's demonstration.
"""

from PIL import Image
from os.path import exists
from timeit import default_timer as timer
import colorsys, os, sys

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

def powerColor(distance, exp, const, scale):
    color = distance**exp
    rgb = colorsys.hsv_to_rgb(const + scale * color, 1 - 0.6 * color, 0.9)
    return tuple(round(i * 255) for i in rgb)

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
                pixels[col, row] = powerColor(distance, 0.2, 0.27, 1.0)

if __name__ == '__main__':
    img = Image.new('RGB', (width, height), color = 'black')
    pixels = img.load()

    s = timer()
    mandel(pixels)
    print("   mandelbrot {:.3f} seconds".format(timer() - s))

    img.save("img1.png")
    print("image saved as img1.png")

  # if sys.platform == "darwin":
  #     os.system("open img1.png")
  # elif sys.platform.startswith("linux") and exists("/usr/bin/eog"):
  #     os.system("eog img1.png")

