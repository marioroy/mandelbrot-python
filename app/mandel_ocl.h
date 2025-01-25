//
// OpenCL common C code for computing the Mandelbrot Set on the CPU or GPU.
//

#if !defined(MANDEL_OCL_H_)
#define MANDEL_OCL_H_ 

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#error "Double-precision floating-point not supported by OpenCL device."
#endif

#if defined(FMA_ON)
#pragma OPENCL FP_CONTRACT ON
#else
#pragma OPENCL FP_CONTRACT OFF
#endif

#define ESCAPE_RADIUS_2 (float) (RADIUS * RADIUS)
#define INSIDE_COLOR1 (uchar4) (0x01,0x01,0x01,0xff)
#define INSIDE_COLOR2 (uchar4) (0x8d,0x02,0x1f,0xff)

#if !defined(M_LN2)
#define M_LN2 0.69314718055994530942  // log_e 2
#endif

typedef union __attribute__((aligned(4))) {
    float f;  // value
    unsigned int i;  // bits
} ufloat_t;

typedef union __attribute__((aligned(8))) {
    double d;  // value
    struct { unsigned int x, y; };  // bits
} udouble_t;

// common functions

static inline uchar4 _get_color(
    __constant const short *colors, double mu )
{
    uchar4 c;
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

static inline bool check_colors(
    const uchar4 c1, const uchar4 c2 )
{
    if (abs((short)c2.x - c1.x) > 8) return true;
    if (abs((short)c2.y - c1.y) > 8) return true;
    if (abs((short)c2.z - c1.z) > 8) return true;

    return false;
}

__kernel void horizontal_gaussian_blur_cpu(
    __constant const float *matrix, __global const uchar4 *src,
    __global uchar4 *dst, const int width, const int height )
{
    int y = get_global_id(0);
    if (y >= height) return;
    const int cols = MATRIX_LENGTH >> 1;

    uchar4 rgb, rgb2;
    float wgt, r, g, b;
    int col2, ix;

    for (int x = 0; x < width; x++) {

        // Gaussian blur optimized 1-D loop.
        rgb = src[mad24(y, width, x)];
        wgt = matrix[cols];
        r = wgt * rgb.x + 0.5f;
        g = wgt * rgb.y + 0.5f;
        b = wgt * rgb.z + 0.5f;
        col2 = cols + cols;

        for (int col = -cols; col < 0; col++) {
            ix = x + col;
            if (ix < 0)
                ix = 0;
            rgb = src[mad24(y, width, ix)];

            ix = x + col + col2;
            if (ix >= width)
                ix = width - 1;
            rgb2 = src[mad24(y, width, ix)];

            wgt = matrix[cols + col];
            r += wgt * ((short)rgb.x + rgb2.x);
            g += wgt * ((short)rgb.y + rgb2.y);
            b += wgt * ((short)rgb.z + rgb2.z);
            col2 -= 2;
        }

        dst[mad24(y, width, x)] = (uchar4)(r, g, b, 0xff);
    }
}

__kernel void horizontal_gaussian_blur(
    __constant const float *matrix, __global const uchar4 *src,
    __global uchar4 *dst, const int width, const int height )
{
    int y, x;

    if (get_global_size(1) == 1) {
        y = get_global_id(0) / width;
        x = get_global_id(0) % width;
    } else {
        y = get_global_id(1);
        x = get_global_id(0);
    }

    if (y >= height || x >= width) return;
    const int cols = MATRIX_LENGTH >> 1;

    // Gaussian blur optimized 1-D loop.
    uchar4 rgb = src[mad24(y, width, x)], rgb2;
    float wgt = matrix[cols];
    float r = wgt * rgb.x + 0.5f;
    float g = wgt * rgb.y + 0.5f;
    float b = wgt * rgb.z + 0.5f;
    int col2 = cols + cols, ix;

    for (int col = -cols; col < 0; col++) {
        ix = x + col;
        if (ix < 0)
            ix = 0;
        rgb = src[mad24(y, width, ix)];

        ix = x + col + col2;
        if (ix >= width)
            ix = width - 1;
        rgb2 = src[mad24(y, width, ix)];

        wgt = matrix[cols + col];
        r += wgt * ((short)rgb.x + rgb2.x);
        g += wgt * ((short)rgb.y + rgb2.y);
        b += wgt * ((short)rgb.z + rgb2.z);
        col2 -= 2;
    }

    dst[mad24(y, width, x)] = (uchar4)(r, g, b, 0xff);
}

__kernel void vertical_gaussian_blur_cpu(
    __constant const float *matrix, __global const uchar4 *src,
    __global uchar4 *dst, const int width, const int height )
{
    int y = get_global_id(0);
    if (y >= height) return;
    const int cols = MATRIX_LENGTH >> 1;

    uchar4 rgb, rgb2;
    float wgt, r, g, b;
    int col2, iy;

    for (int x = 0; x < width; x++) {

        // Gaussian blur optimized 1-D loop.
        rgb = src[mad24(y, width, x)];
        wgt = matrix[cols];
        r = wgt * rgb.x + 0.5f;
        g = wgt * rgb.y + 0.5f;
        b = wgt * rgb.z + 0.5f;
        col2 = cols + cols;

        for (int col = -cols; col < 0; col++) {
            iy = y + col;
            if (iy < 0)
                iy = 0;
            rgb = src[mad24(iy, width, x)];

            iy = y + col + col2;
            if (iy >= height)
                iy = height - 1;
            rgb2 = src[mad24(iy, width, x)];

            wgt = matrix[cols + col];
            r += wgt * ((short)rgb.x + rgb2.x);
            g += wgt * ((short)rgb.y + rgb2.y);
            b += wgt * ((short)rgb.z + rgb2.z);
            col2 -= 2;
        }

        dst[mad24(y, width, x)] = (uchar4)(r, g, b, 0xff);
    }
}

__kernel void vertical_gaussian_blur(
    __constant const float *matrix, __global const uchar4 *src,
    __global uchar4 *dst, const int width, const int height )
{
    int y, x;

    if (get_global_size(1) == 1) {
        y = get_global_id(0) / width;
        x = get_global_id(0) % width;
    } else {
        y = get_global_id(1);
        x = get_global_id(0);
    }

    if (y >= height || x >= width) return;
    const int cols = MATRIX_LENGTH >> 1;

    // Gaussian blur optimized 1-D loop.
    uchar4 rgb = src[mad24(y, width, x)], rgb2;
    float wgt = matrix[cols];
    float r = wgt * rgb.x + 0.5f;
    float g = wgt * rgb.y + 0.5f;
    float b = wgt * rgb.z + 0.5f;
    int col2 = cols + cols, iy;

    for (int col = -cols; col < 0; col++) {
        iy = y + col;
        if (iy < 0)
            iy = 0;
        rgb = src[mad24(iy, width, x)];

        iy = y + col + col2;
        if (iy >= height)
            iy = height - 1;
        rgb2 = src[mad24(iy, width, x)];

        wgt = matrix[cols + col];
        r += wgt * ((short)rgb.x + rgb2.x);
        g += wgt * ((short)rgb.y + rgb2.y);
        b += wgt * ((short)rgb.z + rgb2.z);
        col2 -= 2;
    }

    dst[mad24(y, width, x)] = (uchar4)(r, g, b, 0xff);
}

__kernel void unsharp_mask_cpu(
    __global const uchar4 *src, __global uchar4 *dst, const int width,
    const int height )
{
    int y = get_global_id(0);
    if (y >= height) return;

    const float percent = 65.0f / 100;
    const int threshold = 0;

    int r, g, b, diff;

    for (int x = 0; x < width; x++) {
        int pixel = mad24(y, width, x);
        uchar4 norm_pixel = src[pixel];
        uchar4 blur_pixel = dst[pixel];

        // Compare in/out pixels, apply sharpening.
        diff = (short)norm_pixel.x - blur_pixel.x;
        r = (abs(diff) > threshold)
            ? min(255, max(0, (int)(diff * percent + norm_pixel.x)))
            : norm_pixel.x;

        diff = (short)norm_pixel.y - blur_pixel.y;
        g = (abs(diff) > threshold)
            ? min(255, max(0, (int)(diff * percent + norm_pixel.y)))
            : norm_pixel.y;

        diff = (short)norm_pixel.z - blur_pixel.z;
        b = (abs(diff) > threshold)
            ? min(255, max(0, (int)(diff * percent + norm_pixel.z)))
            : norm_pixel.z;

        dst[pixel] = (uchar4)(r, g, b, 0xff);
    }
}

__kernel void unsharp_mask(
    __global const uchar4 *src, __global uchar4 *dst, const int width,
    const int height )
{
    int y, x;

    if (get_global_size(1) == 1) {
        y = get_global_id(0) / width;
        x = get_global_id(0) % width;
    } else {
        y = get_global_id(1);
        x = get_global_id(0);
    }

    if (y >= height || x >= width) return;

    const int pixel = mad24(y, width, x);
    const float percent = 65.0f / 100;
    const int threshold = 0;

    uchar4 norm_pixel = src[pixel];
    uchar4 blur_pixel = dst[pixel];
    int r, g, b, diff;

    // Compare in/out pixels, apply sharpening.
    diff = (short)norm_pixel.x - blur_pixel.x;
    r = (abs(diff) > threshold)
        ? min(255, max(0, (int)(diff * percent + norm_pixel.x)))
        : norm_pixel.x;

    diff = (short)norm_pixel.y - blur_pixel.y;
    g = (abs(diff) > threshold)
        ? min(255, max(0, (int)(diff * percent + norm_pixel.y)))
        : norm_pixel.y;

    diff = (short)norm_pixel.z - blur_pixel.z;
    b = (abs(diff) > threshold)
        ? min(255, max(0, (int)(diff * percent + norm_pixel.z)))
        : norm_pixel.z;

    dst[pixel] = (uchar4)(r, g, b, 0xff);
}

#endif /* MANDEL_OCL_H_ */

