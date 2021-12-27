# -*- coding: utf-8 -*-
"""
Common constants and functions for mandel_for and mandel_parfor.
"""

import math
import os

from .base import GRADIENT_LENGTH, RADIUS

os.environ['NUMBA_DISABLE_INTEL_SVML'] = str(1)
os.environ['NUMBA_LOOP_VECTORIZE'] = str(0)
os.environ['NUMBA_SLP_VECTORIZE'] = str(0)
os.environ['NUMBA_OPT'] = str(3)

ESCAPE_RADIUS_2 = RADIUS * RADIUS
INSIDE_COLOR1 = (0x01,0x01,0x01)
INSIDE_COLOR2 = (0x8d,0x02,0x1f)
LOG2 = 0.69314718055994530942

import numba as nb
from numba import njit

@njit('UniTuple(u1,3)(i2[:,:], f8, f8, u4)', nogil=True)
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


@njit('b1(u1[:], u1[:])', nogil=True)
def check_colors(c1, c2):

    # Return false if the colors are within tolerance.
    if abs(nb.types.i2(c2[0]) - c1[0]) > 8: return True
    if abs(nb.types.i2(c2[1]) - c1[1]) > 8: return True
    if abs(nb.types.i2(c2[2]) - c1[2]) > 8: return True

    return False


@njit('UniTuple(u1,3)(i2[:,:], f8, f8, u4)', nogil=True)
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


@njit('UniTuple(u1,3)(i2[:,:], f8, f8, u4)', nogil=True)
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

