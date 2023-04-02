# -*- coding: utf-8 -*-
"""
Provides the base class.
"""

__all__ = ["GRADIENT_LENGTH", "INSIDE_COLOR1", "INSIDE_COLOR2", "RADIUS", "Base"]

import math, os, random, sys
import numpy as np

# Suppress subnormal UserWarnings: since numpy 1.22.
# The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
# The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
# Refer to: https://github.com/numpy/numpy/issues/20895

class _suppress_stderr:
    def __init__(self):
        self._stderr = None
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
    def __exit__(self, *args):
        sys.stderr = self._stderr

with _suppress_stderr():
    np.finfo(np.dtype("float32"))
    np.finfo(np.dtype("float64"))

GRADIENT_LENGTH = 260
INSIDE_COLOR1 = (np.uint8(0x01),np.uint8(0x01),np.uint8(0x01))
INSIDE_COLOR2 = (np.uint8(0x8d),np.uint8(0x02),np.uint8(0x1f))
RADIUS = 16.0

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


class Base(object):

    gaussian_kernel = create_gaussian_blur_1d_kernel(2.0)

    @staticmethod
    def divide_up(dividend, divisor):
        """
        Helper funtion to get the next up value for integer division.
        """
        return dividend // divisor + 1 if dividend % divisor else dividend // divisor


    @staticmethod
    def round_up(gsize, bsize):
        """
        Increase the grid size if the block size does not divide evenly.
        """
        r = gsize % bsize

        return gsize + bsize - r if r else gsize


    def fill_linear(self, palette):
        """
        Fill the colors array using linear interpolation.
        """
        num_colors = palette.shape[0]
        num_colors -= 1

        colors = self.colors
        for i in range(GRADIENT_LENGTH):
            mu = float(i) / GRADIENT_LENGTH
            mu *= num_colors
            i_mu = int(mu)
            dx = mu - i_mu
            c1 = palette[i_mu]
            c2 = palette[i_mu + 1]
            colors[i][0] = int(dx * (c2[0] - c1[0]) + c1[0])
            colors[i][1] = int(dx * (c2[1] - c1[1]) + c1[1])
            colors[i][2] = int(dx * (c2[2] - c1[2]) + c1[2])


    def init_colors(self):
        """
        Pick some colors and fill the colors array.
        """
        # Coloring using fixed values.
        if self.color_scheme == 1:
            # Bright qualitative colour scheme, courtesy of Paul Tol.
            # https://personal.sron.nl/~pault/
            p = np.empty((8,3), dtype=np.ctypeslib.ctypes.c_int16)
            p[0], p[p.shape[0] - 1] = (0,0,0), (8,8,8)
            p[1] = (0x44,0x77,0xaa)
            p[2] = (0x66,0xcc,0xee)
            p[3] = (0x22,0x88,0x33)
            p[4] = (0xcc,0xbb,0x44)
            p[5] = (0xee,0x66,0x77)
            p[6] = (0xaa,0x33,0x77)
            self.fill_linear(p)
        elif self.color_scheme == 2:
            p = np.empty((5,3), dtype=np.ctypeslib.ctypes.c_int16)
            p[0], p[p.shape[0] - 1] = (0,0,0), (8,8,8)
            p[1] = (0x65,0xbf,0xa1)
            p[2] = (0x2d,0x95,0xeb)
            p[3] = (0xee,0x4b,0x2f)
            self.fill_linear(p)
        elif self.color_scheme == 3:
            p = np.empty((5,3), dtype=np.ctypeslib.ctypes.c_int16)
            p[0], p[p.shape[0] - 1] = (0,0,0), (8,8,8)
            p[1] = (0xfb,0xc9,0x65)
            p[2] = (0x65,0xcb,0xca)
            p[3] = (0xf8,0x64,0x64)
            self.fill_linear(p)
        elif self.color_scheme == 4:
            p = np.empty((5,3), dtype=np.ctypeslib.ctypes.c_int16)
            p[0], p[p.shape[0] - 1] = (0,0,0), (8,8,8)
            p[1] = (0x77,0xaa,0xdd)
            p[2] = (0xff,0xaa,0xbb)
            p[3] = (0xaa,0xaa,0x00)
            self.fill_linear(p)

        # Coloring using random values.
        elif self.color_scheme == 5:
            random.seed(13)
            p = np.empty((4,3), dtype=np.ctypeslib.ctypes.c_int16)
            p[0], p[p.shape[0] - 1] = (0,0,0), (8,8,8)

            for i in range(1, p.shape[0] - 1):
                p[i][0] = random.uniform(0.0, 1.0) * 155 + 100
                p[i][1] = random.uniform(0.0, 1.0) * 155 + 100
                p[i][2] = random.uniform(0.0, 1.0) * 155 + 100

            self.fill_linear(p)

        # Coloring using a slight modification of Bernstein polynomials.
        # https://mathworld.wolfram.com/BernsteinPolynomial.html
        elif self.color_scheme == 6:
            colors = self.colors
            for n in range(GRADIENT_LENGTH):
                t = n / GRADIENT_LENGTH
                r = int(9 * (1 - t) * t * t * t * 255)
                g = int(15 * (1 - t) * (1 - t) * t * t * 255)
                b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
                colors[n] = (r,g,b)
        # Same thing in gray.
        else:
            colors = self.colors
            for n in range(GRADIENT_LENGTH):
                t = n / GRADIENT_LENGTH
                r = int(9 * (1 - t) * t * t * t * 255)
                g = int(15 * (1 - t) * (1 - t) * t * t * 255)
                b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
                l = int(r * 0.2126 + g * 0.7152 + b * 0.0722 + 0.5)
                colors[n] = (l,l,l)


    def init_offset(self):
        """
        Compute the real and imag AA offsets, used by mandelbrot2.
        Store the offsets into an array.
        """
        aafactor = self.num_samples
        aareach1 = int(aafactor / 2.0)
        aareach2 = aareach1 + 1 if (aafactor % 2) else aareach1
        aaarea = int(aafactor * aafactor)
        aafactorinv = float(1.0 / aafactor)

        self.offset[:] = 0.0
        i = 0

        for xi in range(-aareach1, aareach2, 1):
            for yi in range(-aareach1, aareach2, 1):
                if (xi | yi) == 0:
                    # Mandelbrot1 renders the initial image.
                    continue
                self.offset[i+0] = xi * aafactorinv
                self.offset[i+1] = yi * aafactorinv
                i += 2

