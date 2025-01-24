#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the CPU or GPU using PyOpenCL.
"""

import io, re, os, sys
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH, RADIUS
from app.interface import WindowPygame

NUM_THREADS = min(os.cpu_count(), max(1, OPT.num_threads))
os.environ['CL_CONFIG_CPU_TBB_NUM_WORKERS'] = str(NUM_THREADS)  # Intel OpenCL CPU
os.environ['CPU_MAX_COMPUTE_UNITS'] = str(NUM_THREADS)          # Old AMD OpenCL CPU

# https://portablecl.org/docs/html/using.html#tuning-pocl-behavior-with-env-variables
os.environ['POCL_CPU_MAX_CU_COUNT'] = str(NUM_THREADS)
os.environ['POCL_AFFINITY'] = "1"

KERNEL_SOURCE = ""

try:
    import pyopencl as cl
except ModuleNotFoundError:
    print("[ERROR] No module named pyopencl")
    sys.exit(1)

class App(WindowPygame):

    def __init__(self, opt):
        global KERNEL_SOURCE
        super().__init__(opt)

        if not os.getenv('PYOPENCL_CTX'):
            print("NOTE: Choose a device with double-precision capabilities.")

        try:
            cl_ctx = cl.create_some_context(interactive=True)
        except KeyboardInterrupt:
            sys.exit(0)

        cl_plat = cl_ctx.devices[0].platform.name.strip()
        cl_name = cl_ctx.devices[0].name.strip()
        cl_type = cl_ctx.devices[0].type

        if re.search(r"^Intel\(R\) UHD Graphics", cl_name):
            print(f"[ERROR] {cl_name} lacks double-precision capabilities.")
            del cl_ctx
            sys.exit(1)

        self.is_cuda = True if re.search("^NVIDIA CUDA", cl_plat.upper()) else False
        self.is_cpu = True if cl_type == cl.device_type.CPU else False

        # Intel(R) FPGA Emulation Platform for OpenCL(TM)
        if re.search("FPGA EMULATION", cl_plat.upper()):
            self.is_cpu = True;

        print("[CPU]" if self.is_cpu else "[GPU]", cl_name)

        filepath = os.path.join(os.path.dirname(__file__), \
            'app', 'mandel_ocl.h').replace(' ', '\\ ')
        with io.open(filepath, 'r', encoding='utf-8') as file:
            KERNEL_SOURCE = file.read()

        if OPT.mixed_prec < 3 or self.is_cpu:
            filepath = os.path.join(os.path.dirname(__file__), \
                'app', 'mandel_ocl_mp12.c').replace(' ', '\\ ')
            with io.open(filepath, 'r', encoding='utf-8') as file:
                KERNEL_SOURCE = KERNEL_SOURCE + file.read()
        else:
            filepath = os.path.join(os.path.dirname(__file__), \
                'app', 'mandel_ocl_mp3.c').replace(' ', '\\ ')
            with io.open(filepath, 'r', encoding='utf-8') as file:
                KERNEL_SOURCE = KERNEL_SOURCE + file.read()

        # Construct memory objects.
        self.colors = np.empty((GRADIENT_LENGTH, 3), dtype=np.ctypeslib.ctypes.c_int16)
        self.init_colors()

        self.offset = np.empty(((9*9-1)*2,), dtype=np.ctypeslib.ctypes.c_double)
        self.init_offset()

        h, w = self.height, self.width
        self.output = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_uint32)
        self.temp = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_uint32)
        self.temp2 = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_uint32)

        # Set chunksize for the CPU.
        blur_num_threads = max(1, min(NUM_THREADS, self.divide_up(h, 20)))
        self.chunksize_h = self.divide_up(h, blur_num_threads)

        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        # Allocate OpenCL variables.
        # FMA is determined by the FMA_ON or FMA_OFF switch.
        options = [
            '-D', 'FMA_ON' if self.fma and not self.is_cpu else 'FMA_OFF',
            '-D', 'RADIUS={}'.format(RADIUS),
            '-D', 'GRADIENT_LENGTH={}'.format(GRADIENT_LENGTH),
            '-D', 'MATRIX_LENGTH={}'.format(self.gaussian_kernel.shape[0]) ]

        if self.is_cpu:
            options.extend(['-D', 'MIXED_PREC{}'.format(0)])
        else:
            options.extend(['-D', 'MIXED_PREC{}'.format(self.mixed_prec)])

        try:
            self.cl_prg = cl.Program(cl_ctx, KERNEL_SOURCE).build(options=options)
        except Exception as e:
            print("[ERROR] pyopencl:", e)
            del cl_ctx
            sys.exit(1)

        self.cl_ctx = cl_ctx
        self.cl_queue = cl.CommandQueue(cl_ctx)

        mf = cl.mem_flags
        mf_options_rw = mf.READ_WRITE | mf.USE_HOST_PTR
        mf_options_ro = mf.READ_ONLY | mf.USE_HOST_PTR

        if self.is_cpu:
            self.d_temp = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.temp)
            self.d_temp2 = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.temp2)
            self.d_output = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.output)
            self.d_offset = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.offset)
            self.d_matrix = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.gaussian_kernel)
            self.d_colors = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.colors)
        else:
            self.d_temp = cl.Buffer(cl_ctx, mf.READ_WRITE, self.temp.nbytes)
            self.d_temp2 = cl.Buffer(cl_ctx, mf.READ_WRITE, self.temp2.nbytes)
            self.d_output = cl.Buffer(cl_ctx, mf.READ_WRITE, self.output.nbytes)
            self.d_offset = cl.Buffer(cl_ctx, mf.READ_ONLY, self.offset.nbytes)
            self.d_matrix = cl.Buffer(cl_ctx, mf.READ_ONLY, self.gaussian_kernel.nbytes)
            self.d_colors = cl.Buffer(cl_ctx, mf.READ_ONLY, self.colors.nbytes)
            cl.enqueue_copy(self.cl_queue, self.d_matrix, self.gaussian_kernel).wait()

        # Instantiate the Window interface.
        super().init()

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )

        cl_prg = self.cl_prg
        cl_queue = self.cl_queue

        if self.is_cpu:
            bDimX = 32
            bDimY = 1
            gDimX = self.round_up(self.width * self.height, bDimX)
            gDimY = 1
        else:
            bDimX = 8
            bDimY = 4 if self.is_cuda else 8
            gDimX = self.round_up(self.width, bDimX)
            gDimY = self.round_up(self.height, bDimY)

            if self.update_flag:
                cl.enqueue_copy(cl_queue, self.d_offset, self.offset).wait()
                cl.enqueue_copy(cl_queue, self.d_colors, self.colors).wait()
                self.update_flag = False

        # State 1.
        cl_prg.mandelbrot1(
            cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            np.float64(self.min_x), np.float64(self.min_y),
            np.float64(step_x), np.float64(step_y), self.d_temp, self.d_colors,
            np.int32(iters), np.int32(self.width), np.int32(self.height) )

        cl_queue.finish()

        if self.num_samples == 1:
            if not self.is_cpu:
                cl.enqueue_copy(cl_queue, self.temp, self.d_temp).wait()
            self.update_window()
            return

        # State 2.
        cl_prg.mandelbrot2(
            cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            np.float64(self.min_x), np.float64(self.min_y),
            np.float64(step_x), np.float64(step_y), self.d_output, self.d_temp,
            self.d_colors, np.int32(iters), np.int32(self.width),
            np.int32(self.height), np.int16(self.num_samples), self.d_offset )

        cl_queue.finish()

        # Image sharpening.
        if self.is_cpu:
            self.temp[:] = self.output[:]

            bDimX = self.chunksize_h
            gDimX = self.round_up(self.height, bDimX)

            cl_prg.horizontal_gaussian_blur_cpu(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_matrix, self.d_output, self.d_temp2,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()

            cl_prg.vertical_gaussian_blur_cpu(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_matrix, self.d_temp2, self.d_output,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()

            cl_prg.unsharp_mask_cpu(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_temp, self.d_output,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()

        else:
            cl.enqueue_copy(cl_queue, self.d_temp, self.d_output).wait()

            bDimX = 24
            gDimX = self.round_up(self.width, bDimX)

            cl_prg.horizontal_gaussian_blur(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_matrix, self.d_output, self.d_temp2,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()

            cl_prg.vertical_gaussian_blur(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_matrix, self.d_temp2, self.d_output,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()

            cl_prg.unsharp_mask(
                cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                self.d_temp, self.d_output,
                np.int32(self.width), np.int32(self.height) )

            cl_queue.finish()
            cl.enqueue_copy(cl_queue, self.output, self.d_output).wait()

        self.update_window()

    def exit(self):

        if self.d_colors: self.d_colors.release()
        if self.d_matrix: self.d_matrix.release()
        if self.d_offset: self.d_offset.release()
        if self.d_output: self.d_output.release()
        if self.d_temp: self.d_temp.release()
        if self.d_temp2: self.d_temp2.release()

        del self.cl_queue, self.cl_prg, self.cl_ctx
        del self.colors, self.offset, self.output
        del self.temp, self.temp2


if __name__ == '__main__':

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

