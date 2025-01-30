//
// CUDA C code for computing the Mandelbrot Set on the GPU.
//
// Optimization flag defined in ../mandel_cuda.py:
//   MIXED_PREC3 float-float precision arithmetic and FMA (see dfloat.h)
//   Pixelated near the end of the 64-bit range (--location 1)
//
// mixed-prec=3 1st sample double, supersampling float-float precision
// mixed-prec=4 1st sample and supersampling float-float precision
//
// NVIDIA GeForce RTX 4070 Ti SUPER CUDA results (press x to start auto zoom)
//   ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=3  # 3.6 secs
//   ./mandel_cuda.py --width=1600 --height=900 --mixed_prec=4  # 2.4 secs
//

// #include "dfloat.h"       /* included by ../mandel_cuda.py */
// #include "mandel_cuda.h"  /* included by ../mandel_cuda.py */

// functions

__device__ uchar3 get_color_d(
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

__device__ uchar3 get_color_mp(
    const short *colors, const dfloat_t zreal_sqr2,
    const dfloat_t zimag_sqr2, const int n )
{
    dfloat_t normz2 = sqrt_dfloat(add_dfloat(zreal_sqr2, zimag_sqr2));
    double mu;

    if (RADIUS > 2.0)
        mu = n + (log(2*log(RADIUS)) - log(log(get_dfloat_val(normz2)))) / M_LN2;
    else
        mu = n + 0.5 - log(log(get_dfloat_val(normz2))) / M_LN2;

    return _get_color(colors, mu);
}

__device__ uchar3 mandel1_d(
    const short *colors, const double creal, const double cimag,
    const int max_iters )
{
    udouble_t zreal, zimag;
    udouble_t zreal_sqr, zimag_sqr;
    ufloat_t a, b;

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

            a.i = (zreal_sqr.y & 0xc0000000) | ((zreal_sqr.y & 0x7ffffff) << 3);
            b.i = (zimag_sqr.y & 0xc0000000) | ((zimag_sqr.y & 0x7ffffff) << 3);
            if (a.f + b.f > ESCAPE_RADIUS_2) {
                // Compute 2 more iterations to decrease the error term.
                // http://linas.org/art-gallery/escape/escape.html
                for (int i = 0; i < 2; i++) {
                    zimag.d = fma(2.0 * zreal.d, zimag.d, cimag);
                    zreal.d = zreal_sqr.d - zimag_sqr.d + creal;
                    zreal_sqr.d = zreal.d * zreal.d;
                    zimag_sqr.d = zimag.d * zimag.d;
                }

                return get_color_d(colors, zreal_sqr.d, zimag_sqr.d, n + 3);
            }

            zimag.d = fma(2.0 * zreal.d, zimag.d, cimag);
            zreal.d = zreal_sqr.d - zimag_sqr.d + creal;

            // If the values are equal, than we are in a periodic loop.
            // If not, the outer loop will save the new values and double
            // the number of iterations to test with it.

            if ( (zreal.x == sreal.x) && (zreal.y == sreal.y) &&
                 (zimag.x == simag.x) && (zimag.y == simag.y) ) {
                return INSIDE_COLOR1;
            }

            n += 1;
        }

    } while (n_total != max_iters);

    return INSIDE_COLOR1;
}

__device__ uchar3 mandel1_mp(
    const short *colors, const double creal, const double cimag,
    const int max_iters )
{
    double zreal;

    // Main cardioid bulb test.
    zreal = hypot(creal - 0.25, cimag);
    if (creal < zreal - 2.0 * zreal * zreal + 0.25)
        return INSIDE_COLOR2;

    // Period-2 bulb test to the left of the cardioid.
    zreal = creal + 1.0;
    if (zreal * zreal + cimag * cimag < 0.0625)
        return INSIDE_COLOR2;

    // Periodicity checking i.e. escape early if we detect repetition.
    // http://locklessinc.com/articles/mandelbrot/
    dfloat_t sreal2, simag2;
    int n = 0;
    int n_total = 8;

    dfloat_t two2 = make_dfloat(2.0);
    dfloat_t zreal_sqr2, zimag_sqr2;
    dfloat_t creal2 = make_dfloat(creal);
    dfloat_t cimag2 = make_dfloat(cimag);
    dfloat_t zreal2 = creal2;
    dfloat_t zimag2 = cimag2;
    dfloat_t ztemp1, ztemp2;

    do {
        // Save values to test against.
        sreal2 = zreal2, simag2 = zimag2;

        // Test the next n iterations against those values.
        n_total += n_total;
        if (n_total > max_iters)
            n_total = max_iters;

        // Compute z = z^2 + c.
        while (n < n_total) {
            // zreal_sqr = zreal * zreal;
            zreal_sqr2 = mul_dfloat(zreal2, zreal2);
            // zimag_sqr = zimag * zimag;
            zimag_sqr2 = mul_dfloat(zimag2, zimag2);

            if (get_dfloat_head(zreal_sqr2) + get_dfloat_head(zimag_sqr2) > ESCAPE_RADIUS_2) {
                // Compute 2 more iterations to decrease the error term.
                // http://linas.org/art-gallery/escape/escape.html
                for (int i = 0; i < 2; i++) {
                    // zimag = 2.0 * zreal * zimag + cimag;
                    ztemp1 = mul_dfloat(two2, zreal2);
                    ztemp2 = mul_dfloat(ztemp1, zimag2);
                    zimag2 = add_dfloat(ztemp2, cimag2);

                    // zreal = zreal_sqr - zimag_sqr + creal;
                    ztemp2 = sub_dfloat(zreal_sqr2, zimag_sqr2);
                    zreal2 = add_dfloat(ztemp2, creal2);

                    // zreal_sqr = zreal * zreal;
                    zreal_sqr2 = mul_dfloat(zreal2, zreal2);
                    // zimag_sqr = zimag * zimag;
                    zimag_sqr2 = mul_dfloat(zimag2, zimag2);
                }

                return get_color_mp(colors, zreal_sqr2, zimag_sqr2, n + 3);
            }

            // zimag = 2.0 * zreal * zimag + cimag;
            ztemp1 = mul_dfloat(two2, zreal2);
            ztemp2 = mul_dfloat(ztemp1, zimag2);
            zimag2 = add_dfloat(ztemp2, cimag2);

            // zreal = zreal_sqr - zimag_sqr + creal;
            ztemp2 = sub_dfloat(zreal_sqr2, zimag_sqr2);
            zreal2 = add_dfloat(ztemp2, creal2);

            // If the values are equal, than we are in a periodic loop.
            // If not, the outer loop will save the new values and double
            // the number of iterations to test with it.
            if ( (zreal2.x == sreal2.x) && (zimag2.x == simag2.x) &&
                 (zreal2.y == sreal2.y) && (zimag2.y == simag2.y) ) {
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

  #if defined(MIXED_PREC3)
    // Double Precision arithmetic, FMA
    temp[__umul24(width, pos_y) + pos_x] = mandel1_d(colors, creal, cimag, max_iters);
  #else
    // Float-Float Precision Arithmetic, FMA
    temp[__umul24(width, pos_y) + pos_x] = mandel1_mp(colors, creal, cimag, max_iters);
  #endif
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

    double zreal;
    dfloat_t two2 = make_dfloat(2.0);
    dfloat_t zreal_sqr2, zimag_sqr2;
    dfloat_t ztemp1, ztemp2;
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

        outside = false;

        dfloat_t creal2 = make_dfloat(creal);
        dfloat_t cimag2 = make_dfloat(cimag);
        dfloat_t zreal2 = creal2;
        dfloat_t zimag2 = cimag2;

        for (n = 0; n < max_iters; n++) {
            // zreal_sqr = zreal * zreal;
            zreal_sqr2 = mul_dfloat(zreal2, zreal2);
            // zimag_sqr = zimag * zimag;
            zimag_sqr2 = mul_dfloat(zimag2, zimag2);

            if (get_dfloat_head(zreal_sqr2) + get_dfloat_head(zimag_sqr2) > ESCAPE_RADIUS_2) {
                outside = true;
                break;
            }

            // zimag = 2.0 * zreal * zimag + cimag;
            ztemp1 = mul_dfloat(two2, zreal2);
            ztemp2 = mul_dfloat(ztemp1, zimag2);
            zimag2 = add_dfloat(ztemp2, cimag2);

            // zreal = zreal_sqr - zimag_sqr + creal;
            ztemp2 = sub_dfloat(zreal_sqr2, zimag_sqr2);
            zreal2 = add_dfloat(ztemp2, creal2);
        }

        if (outside) {
            // Compute 2 more iterations to decrease the error term.
            // http://linas.org/art-gallery/escape/escape.html
            for (int i = 0; i < 2; i++) {
                // zimag = 2.0 * zreal * zimag + cimag;
                ztemp1 = mul_dfloat(two2, zreal2);
                ztemp2 = mul_dfloat(ztemp1, zimag2);
                zimag2 = add_dfloat(ztemp2, cimag2);

                // zreal = zreal_sqr - zimag_sqr + creal;
                ztemp2 = sub_dfloat(zreal_sqr2, zimag_sqr2);
                zreal2 = add_dfloat(ztemp2, creal2);

                // zreal_sqr = zreal * zreal;
                zreal_sqr2 = mul_dfloat(zreal2, zreal2);
                // zimag_sqr = zimag * zimag;
                zimag_sqr2 = mul_dfloat(zimag2, zimag2);
            }
            color = get_color_mp(colors, zreal_sqr2, zimag_sqr2, n + 3);
        }
        else
            color = INSIDE_COLOR1;

        c.x += color.x;
        c.y += color.y;
        c.z += color.z;
    }

    output[pixel] = make_uchar3(c.x/aaarea, c.y/aaarea, c.z/aaarea);
}

