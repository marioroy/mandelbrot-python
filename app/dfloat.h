/*
 * CUDA Float-Float Precision Arithmetic release 1.0 (dfloat.h).
 * Copyright (c) 2025 Mario Roy
 *
 * dfloat.h: C/C++ header for computing double precision using dual floats.
 * It enables operations with at least 48-bit significand (2x24-bit) and 8-bit
 * exponent, compared to 53-bit and 11-bit, respectively, of the IEEE 754-1985
 * double precision floating-point format.
 *
 * Based on CUDA Double-Double Precision Arithmetic release 1.2 (dbldbl.h).
 * https://developer.nvidia.com/computeworks-developer-exclusive-downloads
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(DFLOAT_H_)
#define DFLOAT_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#include <math.h>       /* import sqrt() */

/* The head of a float-float number is stored in the most significant part 
   of a float2 (the y-component). The tail is stored in the least significant
   part of the float2 (the x-component). All float-float operands must be 
   normalized on both input to and return from all basic operations, i.e. the
   magnitude of the tail shall be <= 0.5 ulp of the head.
*/
typedef float2 dfloat_t;

/* Create a float-float from a double number */
__device__ __forceinline__ dfloat_t make_dfloat (const double a)
{
    dfloat_t z;
    z.y = __double2float_rn(a); /* head */
    z.x = (float)(a - z.y); /* tail */
    return z;
}

/* Create a float-float from two float numbers */
__device__ __forceinline__ dfloat_t make_dfloat2 (const float head, const float tail)
{
    dfloat_t z;
    z.y = head;
    z.x = tail;
    return z;
}

/* Return the double value of a float-float number */
__device__ __forceinline__ double get_dfloat_val (const dfloat_t a)
{
    return (double)(a.y) + a.x;
}

/* Return the head of a float-float number */
__device__ __forceinline__ float get_dfloat_head (const dfloat_t a)
{
    return a.y;
}

/* Return the tail of a float-float number */
__device__ __forceinline__ float get_dfloat_tail (const dfloat_t a)
{
    return a.x;
}

/* Compute error-free sum of two unordered floats */
__device__ __forceinline__ dfloat_t add_float_to_dfloat (const float a, const float b)
{
    float t1, t2;
    dfloat_t z;
    z.y = __fadd_rn (a, b);
    t1 = __fadd_rn (z.y, -a);
    t2 = __fadd_rn (z.y, -t1);
    t1 = __fadd_rn (b, -t1);
    t2 = __fadd_rn (a, -t2);
    z.x = __fadd_rn (t1, t2);
    return z;
}

/* Compute error-free product of two floats */
__device__ __forceinline__ dfloat_t mul_float_to_dfloat (const float a, const float b)
{
    dfloat_t z;
    z.y = __fmul_rn (a, b);
    z.x = __fmaf_rn (a, b, -z.y);
    return z;
}

/* Negate a float-float number, by separately negating head and tail */
__device__ __forceinline__ dfloat_t neg_dfloat (const dfloat_t a)
{
    dfloat_t z;
    z.y = -a.y;
    z.x = -a.x;
    return z;
}

/* Compute high-accuracy sum of two float-float operands */
__device__ __forceinline__ dfloat_t add_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn (a.y, b.y);
    t2 = __fadd_rn (t1, -a.y);
    t3 = __fadd_rn (__fadd_rn (a.y, t2 - t1), __fadd_rn (b.y, -t2));
    t4 = __fadd_rn (a.x, b.x);
    t2 = __fadd_rn (t4, -a.x);
    t5 = __fadd_rn (__fadd_rn (a.x, t2 - t4), __fadd_rn (b.x, -t2));
    t3 = __fadd_rn (t3, t4);
    t4 = __fadd_rn (t1, t3);
    t3 = __fadd_rn (t1 - t4, t3);
    t3 = __fadd_rn (t3, t5);
    z.y = e = __fadd_rn (t4, t3);
    z.x = __fadd_rn (t4 - e, t3);
    return z;
}

/* Compute high-accuracy difference of two float-float operands */
__device__ __forceinline__ dfloat_t sub_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn (a.y, -b.y);
    t2 = __fadd_rn (t1, -a.y);
    t3 = __fadd_rn (__fadd_rn (a.y, t2 - t1), - __fadd_rn (b.y, t2));
    t4 = __fadd_rn (a.x, -b.x);
    t2 = __fadd_rn (t4, -a.x);
    t5 = __fadd_rn (__fadd_rn (a.x, t2 - t4), - __fadd_rn (b.x, t2));
    t3 = __fadd_rn (t3, t4);
    t4 = __fadd_rn (t1, t3);
    t3 = __fadd_rn (t1 - t4, t3);
    t3 = __fadd_rn (t3, t5);
    z.y = e = __fadd_rn (t4, t3);
    z.x = __fadd_rn (t4 - e, t3);
    return z;
}

/* Compute high-accuracy product of two float-float operands */
__device__ __forceinline__ dfloat_t mul_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t t, z;
    float e;
    t.y = __fmul_rn (a.y, b.y);
    t.x = __fmaf_rn (a.y, b.y, -t.y);
    t.x = __fmaf_rn (a.x, b.x, t.x);
    t.x = __fmaf_rn (a.y, b.x, t.x);
    t.x = __fmaf_rn (a.x, b.y, t.x);
    z.y = e = __fadd_rn (t.y, t.x);
    z.x = __fadd_rn (t.y - e, t.x);
    return z;
}

/* Compute high-accuracy quotient of two float-float operands */
__device__ __forceinline__ dfloat_t div_dfloat (const dfloat_t a, const dfloat_t b)
{
    dfloat_t t, z;
    float e, r;
    r = 1.0f / b.y;
    t.y = __fmul_rn (a.y, r);
    e = __fmaf_rn (b.y, -t.y, a.y);
    t.y = __fmaf_rn (r, e, t.y);
    t.x = __fmaf_rn (b.y, -t.y, a.y);
    t.x = __fadd_rn (a.x, t.x);
    t.x = __fmaf_rn (b.x, -t.y, t.x);
    e = __fmul_rn (r, t.x);
    t.x = __fmaf_rn (b.y, -e, t.x);
    t.x = __fmaf_rn (r, t.x, e);
    z.y = e = __fadd_rn (t.y, t.x);
    z.x = __fadd_rn (t.y - e, t.x);
    return z;
}

/* Compute high-accuracy square root of a float-float number */
__device__ __forceinline__ dfloat_t sqrt_dfloat (const dfloat_t a)
{
    dfloat_t t, z;
    float e, y, s, r;
    r = rsqrtf (a.y);
    if (a.y == 0.0f) r = 0.0f;
    y = __fmul_rn (a.y, r);
    s = __fmaf_rn (y, -y, a.y);
    r = __fmul_rn (0.5f, r);
    z.y = e = __fadd_rn (s, a.x);
    z.x = __fadd_rn (s - e, a.x);
    t.y = __fmul_rn (r, z.y);
    t.x = __fmaf_rn (r, z.y, -t.y);
    t.x = __fmaf_rn (r, z.x, t.x);
    r = __fadd_rn (y, t.y);
    s = __fadd_rn (y - r, t.y);
    s = __fadd_rn (s, t.x);
    z.y = e = __fadd_rn (r, s);
    z.x = __fadd_rn (r - e, s);
    return z;
}

/* Compute high-accuracy reciprocal square root of a float-float number */
__device__ __forceinline__ dfloat_t rsqrt_dfloat (const dfloat_t a)
{
    dfloat_t z;
    float r, s, e;
    r = rsqrtf (a.y);
    e = __fmul_rn (a.y, r);
    s = __fmaf_rn (e, -r, 1.0f);
    e = __fmaf_rn (a.y, r, -e);
    s = __fmaf_rn (e, -r, s);
    e = __fmul_rn (a.x, r);
    s = __fmaf_rn (e, -r, s);
    e = 0.5f * r;
    z.y = __fmul_rn (e, s);
    z.x = __fmaf_rn (e, s, -z.y);
    s = __fadd_rn (r, z.y);
    r = __fadd_rn (r, -s);
    r = __fadd_rn (r, z.y);
    r = __fadd_rn (r, z.x);
    z.y = e = __fadd_rn (s, r);
    z.x = __fadd_rn (s - e, r);
    return z;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* DFLOAT_H_ */
