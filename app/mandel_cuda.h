//
// CUDA common C code for computing the Mandelbrot Set on the GPU.
//

#if !defined(MANDEL_CUDA_H_)
#define MANDEL_CUDA_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ESCAPE_RADIUS_2 (float) (RADIUS * RADIUS)
#define INSIDE_COLOR1 make_uchar3(0x01,0x01,0x01)
#define INSIDE_COLOR2 make_uchar3(0x8d,0x02,0x1f)

#if !defined(M_LN2)
#define M_LN2 0.69314718055994530942  // log_e 2
#endif

typedef union {
    float f;  // value
    unsigned int i;  // bits
} ufloat_t;

typedef union {
    double d;  // value
    struct { unsigned int x, y; };  // bits
} udouble_t;

// common functions

__device__ __forceinline__ uchar3 _get_color(
    const short *colors, double mu )
{
    uchar3 c;
    int i_mu = mu;
    double dx = mu - i_mu;
    int j_mu = dx > 0.0 ? i_mu + 1 : i_mu;

    i_mu = (i_mu % GRADIENT_LENGTH) * 3;
    j_mu = (j_mu % GRADIENT_LENGTH) * 3;

    c.x = dx * (colors[j_mu+0] - colors[i_mu+0]) + colors[i_mu+0];
    c.y = dx * (colors[j_mu+1] - colors[i_mu+1]) + colors[i_mu+1];
    c.z = dx * (colors[j_mu+2] - colors[i_mu+2]) + colors[i_mu+2];

    return c;
}

__device__ __forceinline__ bool check_colors(
    const uchar3 c1, const uchar3 c2 )
{
    if (abs((short)c2.x - c1.x) > 8) return true;
    if (abs((short)c2.y - c1.y) > 8) return true;
    if (abs((short)c2.z - c1.z) > 8) return true;

    return false;
}

__global__ void horizontal_gaussian_blur(
    const float *matrix, const uchar3 *src,
    uchar3 *dst, const int width, const int height )
{
    const int y = __umul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const int x = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (y >= height || x >= width) return;
    const int cols = MATRIX_LENGTH >> 1;

    // Gaussian blur optimized 1-D loop.
    uchar3 rgb = src[__umul24(y, width) + x], rgb2;
    float wgt = matrix[cols];
    float r = wgt * rgb.x + 0.5f;
    float g = wgt * rgb.y + 0.5f;
    float b = wgt * rgb.z + 0.5f;
    int col2 = cols + cols, ix;

    for (int col = -cols; col < 0; col++) {
        ix = x + col;
        if (ix < 0)
            ix = 0;
        rgb = src[__umul24(y, width) + ix];

        ix = x + col + col2;
        if (ix >= width)
            ix = width - 1;
        rgb2 = src[__umul24(y, width) + ix];

        wgt = matrix[cols + col];
        r += wgt * ((short)rgb.x + rgb2.x);
        g += wgt * ((short)rgb.y + rgb2.y);
        b += wgt * ((short)rgb.z + rgb2.z);
        col2 -= 2;
    }

    dst[__umul24(y, width) + x] = make_uchar3(r, g, b);
}

__global__ void vertical_gaussian_blur(
    const float *matrix, const uchar3 *src,
    uchar3 *dst, const int width, const int height )
{
    const int y = __umul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const int x = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (y >= height || x >= width) return;
    const int cols = MATRIX_LENGTH >> 1;

    // Gaussian blur optimized 1-D loop.
    uchar3 rgb = src[__umul24(y, width) + x], rgb2;
    float wgt = matrix[cols];
    float r = wgt * rgb.x + 0.5f;
    float g = wgt * rgb.y + 0.5f;
    float b = wgt * rgb.z + 0.5f;
    int col2 = cols + cols, iy;

    for (int col = -cols; col < 0; col++) {
        iy = y + col;
        if (iy < 0)
            iy = 0;
        rgb = src[__umul24(iy, width) + x];

        iy = y + col + col2;
        if (iy >= height)
            iy = height - 1;
        rgb2 = src[__umul24(iy, width) + x];

        wgt = matrix[cols + col];
        r += wgt * ((short)rgb.x + rgb2.x);
        g += wgt * ((short)rgb.y + rgb2.y);
        b += wgt * ((short)rgb.z + rgb2.z);
        col2 -= 2;
    }

    dst[__umul24(y, width) + x] = make_uchar3(r, g, b);
}

__global__ void unsharp_mask(
    const uchar3 *src, uchar3 *dst, const int width, const int height )
{
    const int y = __umul24(blockDim.y, blockIdx.y) + threadIdx.y;
    const int x = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (y >= height || x >= width) return;

    const int pixel = __umul24(y, width) + x;
    const float percent = 65.0f / 100;
    const int threshold = 0;

    uchar3 norm_pixel = src[pixel];
    uchar3 blur_pixel = dst[pixel];
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

    dst[pixel] = make_uchar3(r, g, b);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* MANDEL_CUDA_H_ */

