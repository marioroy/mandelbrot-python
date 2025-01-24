//
// OpenCL C code for computing the Mandelbrot Set on the CPU or GPU.
// This requires double-precision capabilties on the device.
//
// Optimization flags defined in ../mandel_ocl.py:
//
//   FMA_OFF  GPU matches CPU output (default) i.e. mandel_queue.py.
//   FMA_ON   Enable the Fused-Multiply-Add instruction.
//
//   MIXED_PREC1  integer comparison; this may run faster depending on GPU
//     #if defined(MIXED_PREC1) || defined(MIXED_PREC2)
//       if ( (zreal.x == sreal.x) && (zreal.y == sreal.y) &&
//            (zimag.x == simag.x) && (zimag.y == simag.y) ) {
//     #else
//       if ( (zreal.d == sreal.d) && (zimag.d == simag.d) ) {
//     #endif
//
//   MIXED_PREC2  includes MIXED_PREC1; single-precision addition (hi-bits)
//     #if defined(MIXED_PREC2)
//       a.i = (zreal_sqr.y & 0xc0000000) | ((zreal_sqr.y & 0x7ffffff) << 3);
//       b.i = (zimag_sqr.y & 0xc0000000) | ((zimag_sqr.y & 0x7ffffff) << 3);
//       if (a.f + b.f > ESCAPE_RADIUS_2) {
//     #else
//       if (zreal_sqr.d + zimag_sqr.d > ESCAPE_RADIUS_2) {
//     #endif
//
// Depending on the GPU architecture, mixed_prec=1 may run faster than 2.
// NVIDIA GeForce RTX 4070 Ti SUPER OpenCL results (press x to start auto zoom).
//
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=0 --fma=0  # 6.7 secs
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=0 --fma=1  # 5.4 secs
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=1 --fma=0  # 6.4 secs
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=1 --fma=1  # 5.1 secs
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=2 --fma=0  # 5.7 secs
// ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=2 --fma=1  # 4.8 secs
//


// #include "mandel_ocl.h"  /* included by ../mandel_ocl.py */

#if defined(FMA_ON)
#define _fma(a,b,c) fma(a,b,c)
#else
#define _fma(a,b,c) a * b + c
#endif

#if defined(MIXED_PREC2)
typedef union __attribute__((aligned(4))) {
    float f;  // value
    unsigned int i;  // bits
} ufloat_t;
#endif

typedef union __attribute__((aligned(8))) {
    double d;  // value
    struct { unsigned int x, y; };  // bits
} udouble_t;

// functions

static uchar4 get_color(
    __constant const short *colors, const double zreal_sqr,
    const double zimag_sqr, const int n )
{
    double normz = sqrt(zreal_sqr + zimag_sqr);
    double mu;
    uchar4 c;

    if (RADIUS > 2.0)
        mu = n + (log(2*log(RADIUS)) - log(log(normz))) / M_LN2;
    else
        mu = n + 0.5 - log(log(normz)) / M_LN2;

    int i_mu = mu;
    double dx = mu - i_mu;
    int j_mu = dx > 0.0 ? i_mu + 1 : i_mu;

    i_mu = (i_mu % GRADIENT_LENGTH) * 3;
    j_mu = (j_mu % GRADIENT_LENGTH) * 3;

    c.x = dx * (colors[j_mu+0] - colors[i_mu+0]) + colors[i_mu+0];
    c.y = dx * (colors[j_mu+1] - colors[i_mu+1]) + colors[i_mu+1];
    c.z = dx * (colors[j_mu+2] - colors[i_mu+2]) + colors[i_mu+2];
    c.w = 0xff;

    return c;
}

static uchar4 mandel1(
    __constant const short *colors, const double creal, const double cimag,
    const int max_iters )
{
    udouble_t zreal, zimag;
    udouble_t zreal_sqr, zimag_sqr;
  #if defined(MIXED_PREC2)
    ufloat_t a, b;
  #endif

    // Main cardioid bulb test.
    zreal.d = hypot(creal - 0.25, cimag);
    if (creal < zreal.d - 2.0 * zreal.d * zreal.d + 0.25)
        return INSIDE_COLOR2;

    // Period-2 bulb test to the left of the cardioid.
    zreal.d = creal + 1.0;
    if (zreal.d * zreal.d + cimag * cimag < 0.0625)
        return INSIDE_COLOR2;

    // Periodicity checking i.e. escape early if we detect repetition.
    // http://locklessinc.com/articles/mandelbrot/
    udouble_t sreal, simag;
    int n = 0;
    int n_total = 8;

    zreal.d = creal;
    zimag.d = cimag;

    do {
        // Save values to test against.
        sreal = zreal, simag = zimag;

        // Test the next n iterations against those values.
        n_total += n_total;
        if (n_total > max_iters)
            n_total = max_iters;

        // Compute z = z^2 + c.
        while (n < n_total) {
            zreal_sqr.d = zreal.d * zreal.d;
            zimag_sqr.d = zimag.d * zimag.d;

          #if defined(MIXED_PREC2)
            a.i = (zreal_sqr.y & 0xc0000000) | ((zreal_sqr.y & 0x7ffffff) << 3);
            b.i = (zimag_sqr.y & 0xc0000000) | ((zimag_sqr.y & 0x7ffffff) << 3);
            if (a.f + b.f > ESCAPE_RADIUS_2) {
          #else
            if (zreal_sqr.d + zimag_sqr.d > ESCAPE_RADIUS_2) {
          #endif
                // Compute 2 more iterations to decrease the error term.
                // http://linas.org/art-gallery/escape/escape.html
                for (int i = 0; i < 2; i++) {
                    zimag.d = _fma(2.0 * zreal.d, zimag.d, cimag);
                    zreal.d = zreal_sqr.d - zimag_sqr.d + creal;
                    zreal_sqr.d = zreal.d * zreal.d;
                    zimag_sqr.d = zimag.d * zimag.d;
                }

                return get_color(colors, zreal_sqr.d, zimag_sqr.d, n + 3);
            }

            zimag.d = _fma(2.0 * zreal.d, zimag.d, cimag);
            zreal.d = zreal_sqr.d - zimag_sqr.d + creal;

            // If the values are equal, than we are in a periodic loop.
            // If not, the outer loop will save the new values and double
            // the number of iterations to test with it.

          #if defined(MIXED_PREC1) || defined(MIXED_PREC2)
            if ( (zreal.x == sreal.x) && (zreal.y == sreal.y) &&
                 (zimag.x == simag.x) && (zimag.y == simag.y) ) {
          #else
            if ( (zreal.d == sreal.d) && (zimag.d == simag.d) ) {
          #endif
                return INSIDE_COLOR1;
            }

            n += 1;
        }

    } while (n_total != max_iters);

    return INSIDE_COLOR1;
}

__kernel void mandelbrot1(
    const double min_x, const double min_y, const double step_x,
    const double step_y, __global uchar4 *temp, __constant const short *colors,
    const int max_iters, const int width, const int height )
{
    int pos_y, pos_x;

    if (get_global_size(1) == 1) {
        pos_y = get_global_id(0) / width;
        pos_x = get_global_id(0) % width;
    } else {
        pos_y = get_global_id(1);
        pos_x = get_global_id(0);
    }

    if (pos_y >= height || pos_x >= width) return;

    double cimag = min_y + (pos_y * step_y);
    double creal = min_x + (pos_x * step_x);

    temp[mad24(width, pos_y, pos_x)] = mandel1(colors, creal, cimag, max_iters);
}

__kernel void mandelbrot2(
    const double min_x, const double min_y, const double step_x,
    const double step_y, __global uchar4 *output, __global const uchar4 *temp,
    __constant const short *colors, const int max_iters, const int width,
    const int height, const short aafactor, __constant const double *offset )
{
    int pos_y, pos_x;

    if (get_global_size(1) == 1) {
        pos_y = get_global_id(0) / width;
        pos_x = get_global_id(0) % width;
    } else {
        pos_y = get_global_id(1);
        pos_x = get_global_id(0);
    }

    if (pos_y >= height || pos_x >= width) return;

    int pixel = mad24(width, pos_y, pos_x);
    uchar4 c1 = temp[pixel];
    bool count = false;

    // Skip AA for colors within tolerance.
    if (!count && pos_x > 0)
        count = check_colors(c1, temp[pixel - 1]);
    if (!count && pos_x + 1 < width)
        count = check_colors(c1, temp[pixel + 1]);
    if (!count && pos_x > 1)
        count = check_colors(c1, temp[pixel - 2]);
    if (!count && pos_x + 2 < width)
        count = check_colors(c1, temp[pixel + 2]);

    if (!count && pos_y > 0)
        count = check_colors(c1, temp[pixel - width]);
    if (!count && pos_y + 1 < height)
        count = check_colors(c1, temp[pixel + width]);
    if (!count && pos_y > 1)
        count = check_colors(c1, temp[pixel - width - width]);
    if (!count && pos_y + 2 < height)
        count = check_colors(c1, temp[pixel + width + width]);

    if (!count) {
        output[pixel] = c1;
        return;
    }

    // Compute AA.
    const int aaarea = aafactor * aafactor;
    const int aaarea2 = (aaarea - 1) * 2;
    double creal, cimag;
    int4 c = (int4)(c1.x, c1.y, c1.z, 0xff);
    uchar4 color;

    double zreal, zimag;
    udouble_t zreal_sqr, zimag_sqr;
  #if defined(MIXED_PREC2)
    ufloat_t a, b;
  #endif
    int n;
    bool outside;

    for (int i = 0; i < aaarea2; i += 2) {
        creal = min_x + (((double)pos_x + offset[i]) * step_x);
        cimag = min_y + (((double)pos_y + offset[i+1]) * step_y);

        // Main cardioid bulb test.
        zreal = hypot(creal - 0.25, cimag);
        if (creal < zreal - 2.0 * zreal * zreal + 0.25) {
            color = INSIDE_COLOR2;
            c.x += color.x, c.y += color.y, c.z += color.z;
            continue;
        }

        // Period-2 bulb test to the left of the cardioid.
        zreal = creal + 1.0;
        if (zreal * zreal + cimag * cimag < 0.0625) {
            color = INSIDE_COLOR2;
            c.x += color.x, c.y += color.y, c.z += color.z;
            continue;
        }

        zreal = creal, zimag = cimag;
        outside = false;

        for (n = 0; n < max_iters; n++) {
            zreal_sqr.d = zreal * zreal;
            zimag_sqr.d = zimag * zimag;

          #if defined(MIXED_PREC2)
            a.i = (zreal_sqr.y & 0xc0000000) | ((zreal_sqr.y & 0x7ffffff) << 3);
            b.i = (zimag_sqr.y & 0xc0000000) | ((zimag_sqr.y & 0x7ffffff) << 3);
            if (a.f + b.f > ESCAPE_RADIUS_2) {
          #else
            if (zreal_sqr.d + zimag_sqr.d > ESCAPE_RADIUS_2) {
          #endif
                outside = true;
                break;
            }

            zimag = _fma(2.0 * zreal, zimag, cimag);
            zreal = zreal_sqr.d - zimag_sqr.d + creal;
        }

        if (outside) {
            // Compute 2 more iterations to decrease the error term.
            // http://linas.org/art-gallery/escape/escape.html
            for (int i = 0; i < 2; i++) {
                zimag = _fma(2.0 * zreal, zimag, cimag);
                zreal = zreal_sqr.d - zimag_sqr.d + creal;
                zreal_sqr.d = zreal * zreal;
                zimag_sqr.d = zimag * zimag;
            }
            color = get_color(colors, zreal_sqr.d, zimag_sqr.d, n + 3);
        }
        else
            color = INSIDE_COLOR1;

        c.x += color.x;
        c.y += color.y;
        c.z += color.z;
    }

    output[pixel] = (uchar4)(c.x/aaarea, c.y/aaarea, c.z/aaarea, 0xff);
}

