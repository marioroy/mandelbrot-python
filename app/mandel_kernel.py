# -*- coding: utf-8 -*-
"""
Cuda.jit functions for computing the Mandelbrot Set on the GPU.
This requires double-precision capabilities on the device.

NVIDIA GeForce RTX 2070 cuda.jit result (press x to start auto zoom).
../mandel_kernel.py --width=1280 --height=720 ; 8.1 seconds
"""

import math, os

os.environ['MANDEL_USE_CUDA'] = str(1)

from .base import RADIUS, INSIDE_COLOR1, INSIDE_COLOR2
from .mandel_common import get_color, check_colors, mandel1
from numba import cuda, float32, uint8, int16, int32

ESCAPE_RADIUS_2 = RADIUS * RADIUS

@cuda.jit('void(u1[:,:,:], i2[:,:], i4, i4, f8, f8, f8, f8, i4)')
def mandelbrot1(temp, colors, width, height, min_x, min_y, step_x, step_y, max_iters):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cimag = min_y + (y * step_y)
    creal = min_x + (x * step_x)

    temp[y,x] = mandel1(colors, creal, cimag, max_iters)


@cuda.jit('void(u1[:,:,:], i2[:,:], i4, i4, f8, f8, f8, f8, i4, i4, f8[:], u1[:,:,:])')
def mandelbrot2(temp, colors, width, height, min_x, min_y, step_x, step_y, max_iters, aafactor, offset, output):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

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

    r = int32(c1[0])
    g = int32(c1[1])
    b = int32(c1[2])

    for i in range(0, aaarea2, 2):
        creal = min_x + ((x + offset[i]) * step_x)
        cimag = min_y + ((y + offset[i+1]) * step_y)

        #######################################################
        # Inlined mandel2() for better performance.
        #######################################################

        # Main cardioid bulb test.
        zreal = math.hypot(creal - 0.25, cimag)
        if creal < zreal - 2.0 * zreal * zreal + 0.25:
            r = int32(r + INSIDE_COLOR2[0])
            g = int32(g + INSIDE_COLOR2[1])
            b = int32(b + INSIDE_COLOR2[2])
            continue

        # Period-2 bulb test to the left of the cardioid.
        zreal = creal + 1.0
        if zreal * zreal + cimag * cimag < 0.0625:
            r = int32(r + INSIDE_COLOR2[0])
            g = int32(g + INSIDE_COLOR2[1])
            b = int32(b + INSIDE_COLOR2[2])
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
            zimag = 2.0 * zreal * zimag + cimag
            zreal = zreal_sqr - zimag_sqr + creal

        if outside:
            # Compute 2 more iterations to decrease the error term.
            # http://linas.org/art-gallery/escape/escape.html
            for _ in range(2):
                zimag = 2.0 * zreal * zimag + cimag
                zreal = zreal_sqr - zimag_sqr + creal
                zreal_sqr = zreal * zreal
                zimag_sqr = zimag * zimag

            color = get_color(colors, zreal_sqr, zimag_sqr, n + 3)
        else:
            color = INSIDE_COLOR1

        r = int32(r + color[0])
        g = int32(g + color[1])
        b = int32(b + color[2])

    output[y,x] = (uint8(r/aaarea), uint8(g/aaarea), uint8(b/aaarea))


@cuda.jit('void(f4[:], u1[:,:,:], u1[:,:,:], i4, i4)')
def horizontal_gaussian_blur(matrix, src, dst, width, height):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cols = matrix.shape[0] >> 1

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


@cuda.jit('void(f4[:], u1[:,:,:], u1[:,:,:], i4, i4)')
def vertical_gaussian_blur(matrix, src, dst, width, height):

    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if y >= height or x >= width: return

    cols = matrix.shape[0] >> 1

    # Gaussian blur optimized 1-D loop.
    rgb = src[y,x]
    wgt = matrix[cols]
    r = float32(wgt * rgb[0] + 0.5)
    g = float32(wgt * rgb[1] + 0.5)
    b = float32(wgt * rgb[2] + 0.5)
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
        r += float32(wgt * (int16(rgb[0]) + rgb2[0]))
        g += float32(wgt * (int16(rgb[1]) + rgb2[1]))
        b += float32(wgt * (int16(rgb[2]) + rgb2[2]))
        col2 -= 2

    dst[y,x] = (uint8(r), uint8(g), uint8(b))


@cuda.jit('void(u1[:,:,:], u1[:,:,:], i4, i4)')
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

    percent = float32(65.0 / 100)
    threshold = int16(0)

    # Python version of C code plus multi-core CPU utilization.
    # https://github.com/python-pillow/Pillow/blob/main/src/libImaging/UnsharpMask.c

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

