#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the CPU or GPU using PyOpenCL.
"""

import io, re, os, platform, sys, time
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH, RADIUS
from app.interface import WindowPygame

NUM_THREADS = min(os.cpu_count(), max(1, OPT.num_threads))
os.environ['CL_CONFIG_CPU_TBB_NUM_WORKERS'] = str(NUM_THREADS)  # Intel OpenCL CPU
os.environ['CPU_MAX_COMPUTE_UNITS'] = str(NUM_THREADS)          # Old AMD OpenCL CPU

# Tuning pocl behavior with ENV variables
# https://portablecl.org/docs/html/using.html#tuning-pocl-behavior-with-env-variables
os.environ['POCL_CPU_MAX_CU_COUNT'] = str(NUM_THREADS)
os.environ['POCL_AFFINITY'] = "1"

architecture = platform.machine()
if architecture in ('x86_64', 'AMD64', 'x86'):
    """
    The Intel OpenCL Runtime may not know the host CPU automatically i.e. AMD Ryzen 9800X3D.
    Set environment variable, if empty, to force a SIMD instruction set used for OpenCL
    kernel compilation.
    """
    def get_lscpu_output():
        import subprocess
        try:
            process = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
            return process.stdout
        except subprocess.CalledProcessError as e:
            return f"error executing lscpu: {e}"
        except FileNotFoundError:
            return "lscpu: command not found"

    cpu_target = os.getenv('CL_CONFIG_CPU_TARGET_ARCH', '')
    if not cpu_target and os.name == 'posix' and platform.system() == 'Linux':
        lscpu_output = get_lscpu_output()
        if re.search(r" avx512f", lscpu_output):
            os.environ['CL_CONFIG_CPU_TARGET_ARCH'] = 'skx'
        elif re.search(r" avx2", lscpu_output):
            os.environ['CL_CONFIG_CPU_TARGET_ARCH'] = 'core-avx2'
        elif re.search(r" avx", lscpu_output):
            os.environ['CL_CONFIG_CPU_TARGET_ARCH'] = 'corei7-avx'
        else:
            os.environ['CL_CONFIG_CPU_TARGET_ARCH'] = 'corei7'


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

        if re.search(r"^Clover", cl_plat):
            print(f"[ERROR] The {cl_plat} platform is not supported.")
            del cl_ctx
            sys.exit(1)
        if re.search(r"^Intel\(R\) UHD Graphics", cl_name):
            print(f"[ERROR] The {cl_name} lacks double-precision capabilities.")
            del cl_ctx
            sys.exit(1)
        if re.search(r"^rusticl", cl_plat):
            print(f"[ERROR] The {cl_plat} platform lacks double-precision capabilities.")
            del cl_ctx
            sys.exit(1)

        self.is_cuda = True if re.search("^NVIDIA CUDA", cl_plat.upper()) else False
        self.is_cpu = True if cl_type == cl.device_type.CPU else False

        # Intel(R) FPGA Emulation Platform for OpenCL(TM)
        if re.search("FPGA EMULATION", cl_plat.upper()):
            self.is_cpu = True

        # Determine whether Integrated Graphics Processing Unit
        iGPUs = ('Intel(R) Arc(TM) Graphics', 'gfx1036', 'gfx1037', 'gfx1151')
        self.is_igpu = True if cl_name in iGPUs else False

        if self.is_cpu:
            print("[CPU]", cl_name)
        else:
            print("[GPU]", cl_name, '(integrated)' if self.is_igpu else '(discrete)')

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
                'app', 'mandel_ocl_mp34.c').replace(' ', '\\ ')
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

        if not self.is_cpu:
            self.colr = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_int16)
            self.colg = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_int16)
            self.colb = np.empty((h, w), dtype=np.ctypeslib.ctypes.c_int16)

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
            self.cl_mand1 = cl.Kernel(self.cl_prg, "mandelbrot1")

            if self.is_cpu:
                self.cl_mand2 = cl.Kernel(self.cl_prg, "mandelbrot2_cpu")
                self.cl_hblur = cl.Kernel(self.cl_prg, "horizontal_gaussian_blur_cpu")
                self.cl_vblur = cl.Kernel(self.cl_prg, "vertical_gaussian_blur_cpu")
                self.cl_umask = cl.Kernel(self.cl_prg, "unsharp_mask_cpu")
            else:
                self.cl_mand2 = cl.Kernel(self.cl_prg, "mandelbrot2")
                self.cl_hblur = cl.Kernel(self.cl_prg, "horizontal_gaussian_blur")
                self.cl_vblur = cl.Kernel(self.cl_prg, "vertical_gaussian_blur")
                self.cl_umask = cl.Kernel(self.cl_prg, "unsharp_mask")

        except Exception as e:
            print("[ERROR] pyopencl:", e)
            del cl_ctx
            sys.exit(1)

        self.cl_ctx = cl_ctx
        self.cl_queue = cl.CommandQueue(cl_ctx)

        mf = cl.mem_flags
        mf_options_rw = mf.READ_WRITE | mf.USE_HOST_PTR
        mf_options_ro = mf.READ_ONLY | mf.USE_HOST_PTR

        if self.is_cpu or self.is_igpu:
            self.d_temp = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.temp)
            self.d_temp2 = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.temp2)
            self.d_output = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.output)
            self.d_offset = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.offset)
            self.d_matrix = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.gaussian_kernel)
            self.d_colors = cl.Buffer(cl_ctx, mf_options_ro, hostbuf=self.colors)

            if self.is_igpu:
                self.d_colr = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.colr)
                self.d_colg = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.colg)
                self.d_colb = cl.Buffer(cl_ctx, mf_options_rw, hostbuf=self.colb)
        else:
            self.d_temp = cl.Buffer(cl_ctx, mf.READ_WRITE, self.temp.nbytes)
            self.d_temp2 = cl.Buffer(cl_ctx, mf.READ_WRITE, self.temp2.nbytes)
            self.d_output = cl.Buffer(cl_ctx, mf.READ_WRITE, self.output.nbytes)
            self.d_offset = cl.Buffer(cl_ctx, mf.READ_ONLY, self.offset.nbytes)
            self.d_matrix = cl.Buffer(cl_ctx, mf.READ_ONLY, self.gaussian_kernel.nbytes)
            self.d_colors = cl.Buffer(cl_ctx, mf.READ_ONLY, self.colors.nbytes)
            self.d_colr = cl.Buffer(cl_ctx, mf.READ_WRITE, self.colr.nbytes)
            self.d_colg = cl.Buffer(cl_ctx, mf.READ_WRITE, self.colg.nbytes)
            self.d_colb = cl.Buffer(cl_ctx, mf.READ_WRITE, self.colb.nbytes)

            cl.enqueue_copy(self.cl_queue, self.d_matrix, self.gaussian_kernel).wait()

        # Instantiate the Window interface.
        super().init()

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )

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

            if self.update_flag and not self.is_igpu:
                cl.enqueue_copy(self.cl_queue, self.d_offset, self.offset).wait()
                cl.enqueue_copy(self.cl_queue, self.d_colors, self.colors).wait()
                self.update_flag = False

        # State 1, mandelbrot1.
        self.cl_mand1(
            self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            np.float64(self.min_x), np.float64(self.min_y),
            np.float64(step_x), np.float64(step_y), self.d_temp, self.d_colors,
            np.int32(iters), np.int32(self.width), np.int32(self.height) )

        self.cl_queue.finish()

        if self.num_samples == 1:
            if not (self.is_cpu or self.is_igpu):
                cl.enqueue_copy(self.cl_queue, self.temp, self.d_temp).wait()

            self.update_window()
            return

        # State 2, mandelbrot2.
        if self.is_cpu:
            self.cl_mand2(
                self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                np.float64(self.min_x), np.float64(self.min_y),
                np.float64(step_x), np.float64(step_y), self.d_output, self.d_temp,
                self.d_colors, np.int32(iters), np.int32(self.width),
                np.int32(self.height), np.int16(self.num_samples), self.d_offset )

            self.cl_queue.finish()
        else:
            aaarea = self.num_samples * self.num_samples
            aaarea2 = (aaarea - 1) * 2

            for i in range(0, aaarea2, 2):
                self.cl_mand2(
                    self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
                    np.float64(self.min_x), np.float64(self.min_y),
                    np.float64(step_x), np.float64(step_y), self.d_output,
                    self.d_temp, self.d_colr, self.d_colg, self.d_colb,
                    self.d_colors, np.int32(iters), np.int32(self.width),
                    np.int32(self.height), np.int16(self.num_samples),
                    self.d_offset, np.int32(i) )

                self.cl_queue.finish()
                time.sleep(0.00002) # yield

        # Image sharpening.
        if self.is_cpu:
            self.temp[:] = self.output[:]
            bDimX = self.chunksize_h
            gDimX = self.round_up(self.height, bDimX)
        else:
            cl.enqueue_copy(self.cl_queue, self.d_temp, self.d_output).wait()
            bDimX = 24
            gDimX = self.round_up(self.width, bDimX)

        # Horizontal gaussian blur.
        self.cl_hblur(
            self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            self.d_matrix, self.d_output, self.d_temp2,
            np.int32(self.width), np.int32(self.height) )

        self.cl_queue.finish()

        # Vertical gaussian blur.
        self.cl_vblur(
            self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            self.d_matrix, self.d_temp2, self.d_output,
            np.int32(self.width), np.int32(self.height) )

        self.cl_queue.finish()

        # Unsharp mask.
        self.cl_umask(
            self.cl_queue, (gDimX, gDimY), (bDimX, bDimY),
            self.d_temp, self.d_output,
            np.int32(self.width), np.int32(self.height) )

        self.cl_queue.finish()

        if not self.is_cpu and not self.is_igpu:
            cl.enqueue_copy(self.cl_queue, self.output, self.d_output).wait()

        self.update_window()

    def exit(self):

        if self.d_colors: self.d_colors.release()
        if self.d_matrix: self.d_matrix.release()
        if self.d_offset: self.d_offset.release()
        if self.d_output: self.d_output.release()
        if self.d_temp: self.d_temp.release()
        if self.d_temp2: self.d_temp2.release()

        del self.cl_mand1, self.cl_mand2
        del self.cl_hblur, self.cl_vblur, self.cl_umask
        del self.cl_queue, self.cl_prg, self.cl_ctx
        del self.colors, self.offset, self.output
        del self.temp, self.temp2

        if not self.is_cpu:
            if self.d_colr: self.d_colr.release()
            if self.d_colg: self.d_colg.release()
            if self.d_colb: self.d_colb.release()
            del self.colr, self.colg, self.colb


if __name__ == '__main__':

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

