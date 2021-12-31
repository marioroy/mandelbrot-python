# -*- coding: utf-8 -*-
"""
Mandelbrot functions for cuda.jit.

NVIDIA GeForce RTX 2070 results (press x to start auto zoom).
 cuda.jit
  ../mandel_kernel.py --width=1280 --height=720 ; 8.1 seconds
 PyCUDA
  ../mandel_cuda.py --width=1280 --height=720 --mixed_prec=0 fma=0 ; 9.4 seconds
  ../mandel_cuda.py --width=1280 --height=720 --mixed_prec=0 fma=1 ; 8.0 seconds
  ../mandel_cuda.py --width=1280 --height=720 --mixed_prec=1 fma=0 ; 8.8 seconds
  ../mandel_cuda.py --width=1280 --height=720 --mixed_prec=2 fma=0 ; 7.5 seconds
  ../mandel_cuda.py --width=1280 --height=720 --mixed_prec=2 fma=1 ; 6.7 seconds
"""

import math
import os
import numpy as np
import numba as nb

from numba import cuda

from .base import GRADIENT_LENGTH, RADIUS

ESCAPE_RADIUS_2 = RADIUS * RADIUS
INSIDE_COLOR1 = (0x01,0x01,0x01)
INSIDE_COLOR2 = (0x8d,0x02,0x1f)
LOG2 = 0.69314718055994530942

@cuda.jit(device=True, opt=True)
def get_color(colors, zreal_sqr, zimag_sqr, n):

    # Smooth coloring.
    normz = math.sqrt(zreal_sqr + zimag_sqr)
    if RADIUS > 2.0:
        mu = n + (math.log(2*math.log(RADIUS)) - math.log(math.log(normz))) / LOG2
    else:
        mu = n + 0.5 - math.log(math.log(normz)) / LOG2

    i_mu = int(mu)
    dx = mu - i_mu
    c1 = colors[i_mu % GRADIENT_LENGTH]
    c2 = colors[(i_mu + 1 if dx > 0.0 else i_mu) % GRADIENT_LENGTH]

    r = int(dx * (c2[0] - c1[0]) + c1[0])
    g = int(dx * (c2[1] - c1[1]) + c1[1])
    b = int(dx * (c2[2] - c1[2]) + c1[2])

    return (r,g,b)


@cuda.jit(device=True, opt=True)
def check_colors(c1, c2):

    # Return false if the colors are within tolerance.
    if abs(nb.types.i2(c2[0]) - c1[0]) > 8: return True
    if abs(nb.types.i2(c2[1]) - c1[1]) > 8: return True
    if abs(nb.types.i2(c2[2]) - c1[2]) > 8: return True

    return False


@cuda.jit(device=True, opt=True)
def mandel1(colors, creal, cimag, max_iters):

    # Main cardioid bulb test.
    zreal = math.hypot(creal - 0.25, cimag)
    if creal < zreal - 2 * zreal * zreal + 0.25:
        return INSIDE_COLOR2

    # Period-2 bulb test to the left of the cardioid.
    zreal = creal + 1
    if zreal * zreal + cimag * cimag < 0.0625:
        return INSIDE_COLOR2

    # Periodicity checking i.e. escape early if we detect repetition.
    # http://locklessinc.com/articles/mandelbrot/
    n = 0
    n_total = 8

    zreal = creal
    zimag = cimag

    while True:
        # Save values to test against.
        sreal = zreal
        simag = zimag

        # Test the next n iterations against those values.
        n_total += n_total
        if n_total > max_iters:
            n_total = max_iters

        # Compute z = z^2 + c.
        while n < n_total:
            zreal_sqr = zreal * zreal
            zimag_sqr = zimag * zimag

            if zreal_sqr + zimag_sqr > ESCAPE_RADIUS_2:
                # Compute 2 more iterations to decrease the error term.
                # http://linas.org/art-gallery/escape/escape.html
                for _ in range(2):
                    zimag = 2 * zreal * zimag + cimag
                    zreal = zreal_sqr - zimag_sqr + creal
                    zreal_sqr = zreal * zreal
                    zimag_sqr = zimag * zimag

                return get_color(colors, zreal_sqr, zimag_sqr, n + 3)

            zimag = 2 * zreal * zimag + cimag
            zreal = zreal_sqr - zimag_sqr + creal

            # If the values are equal, than we are in a periodic loop.
            # If not, the outer loop will save the new values and double
            # the number of iterations to test with it.
            if (zreal == sreal) and (zimag == simag):
                return INSIDE_COLOR1

            n += 1

        if n_total == max_iters:
            break

    return INSIDE_COLOR1


@cuda.jit(device=True, opt=True)
def mandel2(colors, creal, cimag, max_iters):

    # Main cardioid bulb test.
    zreal = math.hypot(creal - 0.25, cimag)
    if creal < zreal - 2 * zreal * zreal + 0.25:
        return INSIDE_COLOR2

    # Period-2 bulb test to the left of the cardioid.
    zreal = creal + 1
    if zreal * zreal + cimag * cimag < 0.0625:
        return INSIDE_COLOR2

    zreal = creal
    zimag = cimag

    # Compute z = z^2 + c.
    for n in range(max_iters):
        zreal_sqr = zreal * zreal
        zimag_sqr = zimag * zimag

        if zreal_sqr + zimag_sqr > ESCAPE_RADIUS_2:
            # Compute 2 more iterations to decrease the error term.
            # http://linas.org/art-gallery/escape/escape.html
            for _ in range(2):
                zimag = 2 * zreal * zimag + cimag
                zreal = zreal_sqr - zimag_sqr + creal
                zreal_sqr = zreal * zreal
                zimag_sqr = zimag * zimag

            return get_color(colors, zreal_sqr, zimag_sqr, n + 3)

        zimag = 2 * zreal * zimag + cimag
        zreal = zreal_sqr - zimag_sqr + creal

    return INSIDE_COLOR1


@cuda.jit('void(u1[:,:,:], i2[:,:], i4, i4, f8, f8, f8, f8, i4)', opt=True)
def mandelbrot1(temp, colors, width, height, min_x, min_y, step_x, step_y, max_iters):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cimag = min_y + (y * step_y)
    creal = min_x + (x * step_x)

    temp[y,x] = mandel1(colors, creal, cimag, max_iters)


@cuda.jit('void(u1[:,:,:], i2[:,:], i4, i4, f8, f8, f8, f8, i4, i4, f8[:], u1[:,:,:])', opt=True)
def mandelbrot2(temp, colors, width, height, min_x, min_y, step_x, step_y, max_iters, aafactor, offset, output):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    r = nb.types.i4(0)
    g = nb.types.i4(0)
    b = nb.types.i4(0)

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
        output[y,x] = (c1[0], c1[1], c1[2])
        return

    # Compute AA.
    aaarea = aafactor * aafactor
    aaarea2 = (aaarea - 1) * 2

    r = c1[0]
    g = c1[1]
    b = c1[2]

    for i in range(0, aaarea2, 2):
        creal = min_x + ((x + offset[i]) * step_x)
        cimag = min_y + ((y + offset[i+1]) * step_y)

        #######################################################
        # Inlined mandel2() for better performance.
        #######################################################

        # Main cardioid bulb test.
        zreal = math.hypot(creal - 0.25, cimag)
        if creal < zreal - 2 * zreal * zreal + 0.25:
            r += INSIDE_COLOR2[0]
            g += INSIDE_COLOR2[1]
            b += INSIDE_COLOR2[2]
            continue

        # Period-2 bulb test to the left of the cardioid.
        zreal = creal + 1
        if zreal * zreal + cimag * cimag < 0.0625:
            r += INSIDE_COLOR2[0]
            g += INSIDE_COLOR2[1]
            b += INSIDE_COLOR2[2]
            continue

        zreal = creal
        zimag = cimag
        outside = False

        # Compute z = z^2 + c.
        for n in range(max_iters):
            zreal_sqr = zreal * zreal
            zimag_sqr = zimag * zimag
            if zreal_sqr + zimag_sqr > ESCAPE_RADIUS_2:
                outside = True
                break
            zimag = 2 * zreal * zimag + cimag
            zreal = zreal_sqr - zimag_sqr + creal

        if outside:
            # Compute 2 more iterations to decrease the error term.
            # http://linas.org/art-gallery/escape/escape.html
            for _ in range(2):
                zimag = 2 * zreal * zimag + cimag
                zreal = zreal_sqr - zimag_sqr + creal
                zreal_sqr = zreal * zreal
                zimag_sqr = zimag * zimag

            color = get_color(colors, zreal_sqr, zimag_sqr, n + 3)
        else:
            color = INSIDE_COLOR1

        r += color[0]
        g += color[1]
        b += color[2]

    output[y,x] = (int(r/aaarea), int(g/aaarea), int(b/aaarea))


@cuda.jit('void(f4[:], u1[:,:,:], u1[:,:,:], i4, i4)', opt=True)
def horizontal_gaussian_blur(matrix, src, dst, width, height):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cols = matrix.shape[0] >> 1

    # Gaussian blur optimized 1-D loop.
    rgb = src[y,x]
    wgt = matrix[cols]
    r = nb.types.f4(wgt * rgb[0] + 0.5)
    g = nb.types.f4(wgt * rgb[1] + 0.5)
    b = nb.types.f4(wgt * rgb[2] + 0.5)
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
        r += nb.types.f4(wgt * (nb.types.i2(rgb[0]) + rgb2[0]))
        g += nb.types.f4(wgt * (nb.types.i2(rgb[1]) + rgb2[1]))
        b += nb.types.f4(wgt * (nb.types.i2(rgb[2]) + rgb2[2]))
        col2 -= 2

    dst[y,x] = (int(r), int(g), int(b))


@cuda.jit('void(f4[:], u1[:,:,:], u1[:,:,:], i4, i4)', opt=True)
def vertical_gaussian_blur(matrix, src, dst, width, height):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cols = matrix.shape[0] >> 1

    # Gaussian blur optimized 1-D loop.
    rgb = src[y,x]
    wgt = matrix[cols]
    r = nb.types.f4(wgt * rgb[0] + 0.5)
    g = nb.types.f4(wgt * rgb[1] + 0.5)
    b = nb.types.f4(wgt * rgb[2] + 0.5)
    col2 = cols + cols

    for col in range(-cols, 0):
        iy = y + col
        if iy < 0:
            iy = 0
        rgb = src[iy,x]

        iy = y + col + col2
        if iy >= height:
            iy = height - 1
        rgb2 = src[iy,x]

        wgt = matrix[cols + col]
        r += nb.types.f4(wgt * (nb.types.i2(rgb[0]) + rgb2[0]))
        g += nb.types.f4(wgt * (nb.types.i2(rgb[1]) + rgb2[1]))
        b += nb.types.f4(wgt * (nb.types.i2(rgb[2]) + rgb2[2]))
        col2 -= 2

    dst[y,x] = (int(r), int(g), int(b))


@cuda.jit('void(u1[:,:,:], u1[:,:,:], i4, i4)', opt=True)
def unsharp_mask(src, dst, width, height):
    """
    Sharpen the destination image using the Unsharp Mask technique.

    Compare "normal" pixel from "src" to "blurred" pixel from "dst".
    If the difference is more than threshold value, apply the OPPOSITE
    correction to the amount of blur, multiplied by percent.
    """
    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    percent = nb.types.f4(65.0 / 100)
    threshold = nb.types.i2(0)

    # Python version of C code plus multi-core CPU utilization.
    # https://github.com/python-pillow/Pillow/blob/main/src/libImaging/UnsharpMask.c

    norm_pixel = src[y,x]
    blur_pixel = dst[y,x]

    # Compare in/out pixels, apply sharpening.
    diff = nb.types.i2(nb.types.i2(norm_pixel[0]) - blur_pixel[0])
    if abs(diff) > threshold:
        # Add the difference to the original pixel.
        r = min(255, max(0, int(diff * percent + norm_pixel[0])))
    else:
        # New pixel is the same as the original pixel.
        r = norm_pixel[0]

    diff = nb.types.i2(nb.types.i2(norm_pixel[1]) - blur_pixel[1])
    if abs(diff) > threshold:
        g = min(255, max(0, int(diff * percent + norm_pixel[1])))
    else:
        g = norm_pixel[1]

    diff = nb.types.i2(nb.types.i2(norm_pixel[2]) - blur_pixel[2])
    if abs(diff) > threshold:
        b = min(255, max(0, int(diff * percent + norm_pixel[2])))
    else:
        b = norm_pixel[2]

    dst[y,x] = (r,g,b)

