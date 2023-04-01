#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the Mandelbrot Set on the CPU using queues for IPC.

Preferably use Python 3.7 or later for better performance.
SimpleQueue has noticeably lesser overhead compared to Queue.
See app/parallel.py.
"""

import math, os, sys
import numpy as np

from app.option import OPT
from app.base import GRADIENT_LENGTH
from app.interface import WindowPygame
from app.parallel import USE_FORK, Barrier, Queue, Thread
from multiprocessing import RawArray

from app.mandel_for import \
    mandelbrot1, mandelbrot2, horizontal_gaussian_blur, unsharp_mask

class App(WindowPygame):

    def __init__(self, opt):
        super().__init__(opt)

        self.num_threads = min(opt.num_threads, self.height)
        print("[CPU] number of threads {}".format(self.num_threads))
        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        # Construct shared-memory objects.
        shm_colors = RawArray(np.ctypeslib.ctypes.c_int16, int(GRADIENT_LENGTH*3))
        self.colors = np.ctypeslib.as_array(shm_colors)
        self.colors = self.colors.reshape((GRADIENT_LENGTH, 3))
        self.init_colors()

        shm_offset = RawArray(np.ctypeslib.ctypes.c_double, int((9*9-1) * 2))
        self.offset = np.ctypeslib.as_array(shm_offset)
        self.init_offset()

        h, w = self.height, self.width
        shm_output = RawArray(np.ctypeslib.ctypes.c_uint8, int(h*w*3))
        self.output = np.ctypeslib.as_array(shm_output)
        self.output = self.output.reshape((h, w, 3))

        shm_temp = RawArray(np.ctypeslib.ctypes.c_uint8, int(h*w*3))
        self.temp = np.ctypeslib.as_array(shm_temp)
        self.temp = self.temp.reshape((h, w, 3))

        shm_temp2 = RawArray(np.ctypeslib.ctypes.c_uint8, int(h*w*3))
        self.temp2 = np.ctypeslib.as_array(shm_temp2)
        self.temp2 = self.temp2.reshape((h, w, 3))

        # Spawn workers.
        blur_num_threads = max(1, min(self.num_threads, self.divide_up(h, 20)))

        self.barrier_blur = Barrier(blur_num_threads)
        self.barrier_blur2 = Barrier(blur_num_threads + 1)
        self.barrier_chunk = Barrier(self.num_threads + 1)

        self.queue_job = Queue()
        self.queue_data = Queue()

        self.consumers = list()
        for wid in range(1, self.num_threads + 1):
            args = (wid, self.height, self.width, self.gaussian_kernel)
            self.consumers.append(Thread(target=self.cpu_task, args=args))
            self.consumers[-1].start()

        # Instantiate the Window interface.
        super().init()

    def __cpu_task(self, wid, height, width, kernel):

        blur_num_threads = self.barrier_blur.parties
        chunksize_h = self.divide_up(height, blur_num_threads)
        chunksize_w = self.divide_up(width, blur_num_threads)

        output_T = self.output.swapaxes(0,1)  # transpose(1,0,2)
        temp2_T = self.temp2.swapaxes(0,1)

        # Receive job parameters.
        while True:
            args = self.queue_job.get()
            if args is None: break

            num_chunks, min_x, min_y, step_x, step_y, num_samples, iters, \
                state = args

            # Process chunk data.
            while True:
                args = self.queue_data.get()
                chunk_id, seqH = args
                if seqH:
                    if state == 1:
                        mandelbrot1(
                            self.temp, self.colors, seqH, min_x, min_y, step_x,
                            step_y, iters )
                    else:
                        mandelbrot2(
                            self.temp, self.colors, seqH, min_x, min_y, step_x,
                            step_y, iters, num_samples, self.offset, self.output )

                # Wait for any remaining chunks to finish.
                if chunk_id + self.num_threads > num_chunks:
                    self.barrier_chunk.wait()      # sync including manager

                    # Image sharpening.
                    if state == 2 and wid <= blur_num_threads:
                        start = (wid - 1) * chunksize_h
                        stop = start + chunksize_h
                        seqH = (start, stop if stop <= height else height)

                        start = (wid - 1) * chunksize_w
                        stop = start + chunksize_w
                        seqW = (start, stop if stop <= width else width)

                        self.barrier_blur2.wait()  # wait for the manager to finish copy

                        horizontal_gaussian_blur(kernel, self.output, self.temp2, seqH)
                        self.barrier_blur.wait()   # sync among blur threads
                        horizontal_gaussian_blur(kernel, temp2_T, output_T, seqW)
                        self.barrier_blur.wait()   # sync among blur threads
                        unsharp_mask(self.temp, self.output, seqH)

                        self.barrier_blur2.wait()  # sync including manager

                    break

    def cpu_task(self, wid, height, width, kernel):

      # if USE_FORK and sys.platform == 'linux' and os.cpu_count() > 2:
      #     pid = os.getpid()
      #     cnt = os.cpu_count()
      #     cpu = (wid - 1) % cnt if wid <= cnt - 2 else f"0-{cnt - 1}"
      #     cmd = f"taskset --cpu-list --pid {cpu} {pid} >/dev/null"
      #     os.system(cmd)

        try:
            self.__cpu_task(wid, height, width, kernel)
        except KeyboardInterrupt:
            pass

    def display(self):

        iters, step_x, step_y = self.update_iters( show_info=True )
        chunksize = max(2, int(math.ceil(300 / self.width * 2)))

        for state in range(1, 2 if self.num_samples == 1 else 3):
            if state == 2 and iters >= 600:
                chunksize = chunksize // 2

            # Submit job parameters followed by chunked data.
            num_chunks = self.divide_up(self.height, chunksize)
            args = ( num_chunks, self.min_x, self.min_y, step_x, step_y,
                     self.num_samples, iters, state )

            for _ in range(self.num_threads):
                self.queue_job.put(args)

            for i in range(num_chunks):
                start = i * chunksize
                stop = start + chunksize
                args = (i+1, (start, stop if stop <= self.height else self.height))
                self.queue_data.put(args)

            # Notify available threads to wait.
            if num_chunks < self.num_threads:
                for _ in range(self.num_threads - num_chunks):
                    self.queue_data.put((num_chunks, None))

            self.barrier_chunk.wait()

            # Image sharpening.
            if state == 2:
                self.temp[:] = self.output[:] # make a copy of OUTPUT for unsharp_mask
                self.barrier_blur2.wait()     # workers wait for the copy to finish
                self.barrier_blur2.wait()     # wait for blur & image sharpening to finish

        self.update_window()

    def exit(self):

        for _ in range(len(self.consumers)):
            self.queue_job.put(None)

        for c in self.consumers:
            c.join()

        del self.colors, self.offset, self.output
        del self.temp, self.temp2


if __name__ == '__main__':

  # if USE_FORK and sys.platform == 'linux' and os.cpu_count() > 2:
  #     pid = os.getpid()
  #     cpu1 = os.cpu_count() - 1
  #     cpu2 = os.cpu_count() - 2
  #     cmd = f"taskset --cpu-list --pid {cpu1},{cpu2} {pid} >/dev/null"
  #     os.system(cmd)

    mandel = App(OPT)
    try:
        mandel.run()
        mandel.exit()
    except KeyboardInterrupt:
        mandel.exit()

