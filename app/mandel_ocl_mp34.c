//
// OpenCL C code for computing the Mandelbrot Set on the GPU.
//
// Optimization flag defined in ../mandel_ocl.py:
//   MIXED_PREC3 float-float precision arithmetic and FMA
//   Pixelated near the end of the 64-bit range (--location 1)
//
// mixed-prec=3 1st sample double, supersampling float-float precision
// mixed-prec=4 1st sample and supersampling float-float precision
//
// NVIDIA GeForce RTX 4070 Ti SUPER OpenCL results (press x to start auto zoom)
//   ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=3  # 3.8 secs
//   ./mandel_ocl.py --width=1600 --height=900 --mixed_prec=4  # 2.6 secs
//

// #include "mandel_ocl.h"  /* included by ../mandel_ocl.py */

// float-float precision arithmetic

typedef float2 dfloat_t;

/* Create a float-float from a double number */
static inline dfloat_t make_dfloat (const double a)
{
    dfloat_t z;
    z.y = (float)(a); /* head */
    z.x = (float)(a - z.y); /* tail */
    return z;
}

/* Return the double value of a float-float number */
static inline double get_dfloat_val (const dfloat_t a)
{
    return (double)(a.y) + a.x;
}

/* Return the head of a float-float number */
static inline float get_dfloat_head (const dfloat_t a)
{
    return a.y;
}

/* Return the tail of a float-float number */
static inline float get_dfloat_tail (const dfloat_t a)
{
    return a.x;
}

/* Compute high-accuracy sum of two float-float operands */
static inline dfloat_t add_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t z;
    float t1, t2, t3, t4, t5, e;
    t1 = a.y + b.y;
    t2 = t1 + -a.y;
    t3 = a.y + (t2 - t1) + (b.y + -t2);
    t4 = a.x + b.x;
    t2 = t4 + -a.x;
    t5 = a.x + (t2 - t4) + (b.x + -t2);
    t3 = t3 + t4;
    t4 = t1 + t3;
    t3 = t1 - t4 + t3;
    t3 = t3 + t5;
    z.y = e = t4 + t3;
    z.x = t4 - e + t3;
    return z;
}

/* Compute high-accuracy difference of two float-float operands */
static inline dfloat_t sub_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t z;
    float t1, t2, t3, t4, t5, e;
    t1 = a.y + -b.y;
    t2 = t1 + -a.y;
    t3 = a.y + (t2 - t1) + -(b.y + t2);
    t4 = a.x + -b.x;
    t2 = t4 + -a.x;
    t5 = a.x + (t2 - t4) + -(b.x + t2);
    t3 = t3 + t4;
    t4 = t1 + t3;
    t3 = t1 - t4 + t3;
    t3 = t3 + t5;
    z.y = e = t4 + t3;
    z.x = t4 - e + t3;
    return z;
}

/* Compute high-accuracy product of two float-float operands */
static inline dfloat_t mul_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t t, z;
    float e;
    t.y = a.y * b.y;
    t.x = fma (a.y, b.y, -t.y);
    t.x = fma (a.x, b.x, t.x);
    t.x = fma (a.y, b.x, t.x);
    t.x = fma (a.x, b.y, t.x);
    z.y = e = t.y + t.x;
    z.x = t.y - e + t.x;
    return z;
}

/* Compute high-accuracy square root of a float-float number */
static inline dfloat_t sqrt_dfloat (const dfloat_t a)
{
    dfloat_t t, z;
    float e, y, s, r;
    r = rsqrt (a.y);
    if (a.y == 0.0f) r = 0.0f;
    y = a.y * r;
    s = fma (y, -y, a.y);
    r = 0.5f * r;
    z.y = e = s + a.x;
    z.x = s - e + a.x;
    t.y = r * z.y;
    t.x = fma (r, z.y, -t.y);
    t.x = fma (r, z.x, t.x);
    r = y + t.y;
    s = y - r + t.y;
    s = s + t.x;
    z.y = e = r + s;
    z.x = r - e + s;
    return z;
}

// functions

static uchar4 get_color_d(
    __constant const short *colors, const double zreal_sqr,
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

static uchar4 get_color_mp(
    __constant const short *colors, const dfloat_t zreal2_sqr,
    const dfloat_t zimag2_sqr, const int n )
{
    dfloat_t normz2 = sqrt_dfloat(add_dfloat(zreal2_sqr, zimag2_sqr));
    double mu;

    if (RADIUS > 2.0)
        mu = n + (log(2*log(RADIUS)) - log(log(get_dfloat_val(normz2)))) / M_LN2;
    else
        mu = n + 0.5 - log(log(get_dfloat_val(normz2))) / M_LN2;

    return _get_color(colors, mu);
}

static uchar4 mandel1_d(
    __constant const short *colors, const double creal, const double cimag,
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

static uchar4 mandel1_mp(
    __constant const short *colors, const double creal, const double cimag,
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
    dfloat_t zreal2_sqr, zimag2_sqr;
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
            zreal2_sqr = mul_dfloat(zreal2, zreal2);
            // zimag_sqr = zimag * zimag;
            zimag2_sqr = mul_dfloat(zimag2, zimag2);

            if (get_dfloat_head(zreal2_sqr) + get_dfloat_head(zimag2_sqr) > ESCAPE_RADIUS_2) {
                // Compute 2 more iterations to decrease the error term.
                // http://linas.org/art-gallery/escape/escape.html
                for (int i = 0; i < 2; i++) {
                    // zimag = 2.0 * zreal * zimag + cimag;
                    ztemp1 = mul_dfloat(two2, zreal2);
                    ztemp2 = mul_dfloat(ztemp1, zimag2);
                    zimag2 = add_dfloat(ztemp2, cimag2);

                    // zreal = zreal_sqr - zimag_sqr + creal;
                    ztemp2 = sub_dfloat(zreal2_sqr, zimag2_sqr);
                    zreal2 = add_dfloat(ztemp2, creal2);

                    // zreal_sqr = zreal * zreal;
                    zreal2_sqr = mul_dfloat(zreal2, zreal2);
                    // zimag_sqr = zimag * zimag;
                    zimag2_sqr = mul_dfloat(zimag2, zimag2);
                }

                return get_color_mp(colors, zreal2_sqr, zimag2_sqr, n + 3);
            }

            // zimag = 2.0 * zreal * zimag + cimag;
            ztemp1 = mul_dfloat(two2, zreal2);
            ztemp2 = mul_dfloat(ztemp1, zimag2);
            zimag2 = add_dfloat(ztemp2, cimag2);

            // zreal = zreal_sqr - zimag_sqr + creal;
            ztemp2 = sub_dfloat(zreal2_sqr, zimag2_sqr);
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

  #if defined(MIXED_PREC3)
    // Double Precision Arithmetic, FMA
    temp[mad24(width, pos_y, pos_x)] = mandel1_d(colors, creal, cimag, max_iters);
  #else
    // Float-Float Precision Arithmetic, FMA
    temp[mad24(width, pos_y, pos_x)] = mandel1_mp(colors, creal, cimag, max_iters);
  #endif
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

    double zreal;
    dfloat_t two2 = make_dfloat(2.0); 
    dfloat_t zreal2_sqr, zimag2_sqr;
    dfloat_t ztemp1, ztemp2;
    int n;
    bool outside;

    for (int i = 0; i < aaarea2; i += 2) {
        creal = min_x + (((double)pos_x + offset[i]) * step_x);
        cimag = min_y + (((double)pos_y + offset[i+1]) * step_y);

        dfloat_t creal2 = make_dfloat(creal);
        dfloat_t cimag2 = make_dfloat(cimag);
        dfloat_t zreal2 = creal2;
        dfloat_t zimag2 = cimag2;

        outside = false;

        for (n = 0; n < max_iters; n++) {
            // zreal_sqr = zreal * zreal;
            zreal2_sqr = mul_dfloat(zreal2, zreal2);
            // zimag_sqr = zimag * zimag;
            zimag2_sqr = mul_dfloat(zimag2, zimag2);

            if (get_dfloat_head(zreal2_sqr) + get_dfloat_head(zimag2_sqr) > ESCAPE_RADIUS_2) {
                outside = true;
                break;
            }

            // zimag = 2.0 * zreal * zimag + cimag;
            ztemp1 = mul_dfloat(two2, zreal2);
            ztemp2 = mul_dfloat(ztemp1, zimag2);
            zimag2 = add_dfloat(ztemp2, cimag2);

            // zreal = zreal_sqr - zimag_sqr + creal;
            ztemp2 = sub_dfloat(zreal2_sqr, zimag2_sqr);
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
                ztemp2 = sub_dfloat(zreal2_sqr, zimag2_sqr);
                zreal2 = add_dfloat(ztemp2, creal2);

                // zreal_sqr = zreal * zreal;
                zreal2_sqr = mul_dfloat(zreal2, zreal2);
                // zimag_sqr = zimag * zimag;
                zimag2_sqr = mul_dfloat(zimag2, zimag2);
            }
            color = get_color_mp(colors, zreal2_sqr, zimag2_sqr, n + 3);
        }
        else
            color = INSIDE_COLOR1;

        c.x += color.x;
        c.y += color.y;
        c.z += color.z;
    }

    output[pixel] = (uchar4)(c.x/aaarea, c.y/aaarea, c.z/aaarea, 0xff);
}

