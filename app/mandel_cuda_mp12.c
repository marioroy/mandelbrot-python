//
// CUDA C code for computing the Mandelbrot Set on the GPU.
//
// Optimization flags defined in ../mandel_cuda.py:
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
// NVIDIA GeForce RTX 4070 Ti SUPER CUDA results (press x to start auto zoom)
//
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=0 --fma=0  # 6.1 secs
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=0 --fma=1  # 5.2 secs
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=1 --fma=0  # 5.8 secs
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=1 --fma=1  # 5.1 secs
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=2 --fma=0  # 5.1 secs
// ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=2 --fma=1  # 4.6 secs
//

// #include "mandel_cuda.h"  // included by ../mandel_cuda.py

#if defined(FMA_ON)
#define _fma(a,b,c) fma(a,b,c)
#else
#define _fma(a,b,c) a * b + c
#endif

// functions

__device__ uchar3 get_color(
    const short *colors, const double zreal_sqr,
    const double zimag_sqr, const int n )
{
    double normz = sqrt(zreal_sqr + zimag_sqr);
    double mu;

    if (RADIUS > 2.0)
        mu = n + (log(2*log(RADIUS)) - log(log(normz))) / M_LN2;
    else
        mu = n + 0.5 - log(log(normz)) / M_LN2;

    return _get_color(colors, mu);
}

__device__ uchar3 mandel1(
    const short *colors, const double creal, const double cimag,
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

__global__ void mandelbrot1(
    const double min_x, const double min_y, const double step_x,
    const double step_y, uchar3 *temp, const short *colors,
    const int max_iters, const int width, const int height )
{
    const int pos_y = __umul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const int pos_x = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (pos_y >= height || pos_x >= width) return;

    double cimag = min_y + (pos_y * step_y);
    double creal = min_x + (pos_x * step_x);

    temp[__umul24(width, pos_y) + pos_x] = mandel1(colors, creal, cimag, max_iters);
}

__global__ void mandelbrot2(
    const double min_x, const double min_y, const double step_x,
    const double step_y, uchar3 *output, const uchar3 *temp,
    const short *colors, const int max_iters, const int width,
    const int height, const short aafactor, const double *offset )
{
    const int pos_y = __umul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const int pos_x = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (pos_y >= height || pos_x >= width) return;

    int pixel = __umul24(width, pos_y) + pos_x;
    uchar3 c1 = temp[pixel];
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
    int3 c = make_int3(c1.x, c1.y, c1.z);
    uchar3 color;

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

    output[pixel] = make_uchar3(c.x/aaarea, c.y/aaarea, c.z/aaarea);
}

