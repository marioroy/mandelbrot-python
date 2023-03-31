# -*- coding: utf-8 -*-
"""
Provides the Pygame-based window interface.
"""

__all__ = ["WindowPygame"]

import math, os, sys, time
os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1';

import numpy as np
import pygame as pg

from .base import Base

class WindowPygame(Base):

    def __init__(self, opt):

        self.width = opt.width
        self.height = opt.height
        self.color_scheme = opt.color_scheme
        self.num_samples = opt.num_samples
        self.perf_level = opt.perf_level
        self.fast_zoom = opt.fast_zoom
        self.smooth_bench = opt.smooth_bench
        self.mixed_prec = opt.mixed_prec
        self.fma = opt.fma
        self.magf = 0.887168 if self.fast_zoom else 0.9332
        self.perf_win = int(math.ceil(self.width / 800 * self.perf_level))
        self.rgb_sum = 0
        self.rgb_sum_reset = True
        self.update_flag = True # used by OpenCL and CUDA
        self.num_clicks = 0
        self.start_click = 0.0
        self.scroll_factor = 0.1
        self.level = 0

        # init/save home location
        self.center_x, self.center_y, self.zoom_scale = \
            opt.center_x, opt.center_y, opt.zoom_scale

        self.update_level() # call first before update_iters
        self.update_iters()

        self.reset_view = (
            self.level, self.center_x, self.center_y,
            self.iters, self.zoom_scale )

        # begin at location if specified
        if opt.location:
            switch = {
              1: [-1.7485854655160802,  0.0126275661380999, 10042274467041.889],
              2: [-0.7746806106269039, -0.1374168856037867, 1422616677257.3857],
              3: [-0.0801706662109324,  0.6568767849265752, 3068340407.9648294],
              4: [-1.7689786824401361,  0.0046053975000913, 29868296.757352185],
              5: [-0.4391269761085219,  0.5745831678340557, 3060043.2274685167],
              6: [-0.7426799200664,     0.16382294543399,   7273.1447566685370],
              7: [-0.5476366631259864,  0.627355515204217,  256749985.20257786],
              }
            default = [self.center_x, self.center_y, self.zoom_scale]
            self.center_x, self.center_y, self.zoom_scale = \
                switch.get(opt.location, default)

            self.update_level()
            self.update_iters()


    def init(self):

        # There's no sound or anything like that. Thus initializing display only.
        pg.display.init()

        self.window = pg.display.set_mode((self.width, self.height), flags=pg.DOUBLEBUF|pg.HIDDEN)
        self.window.set_alpha(None)
        self.window.fill(pg.Color('#000000'))

        pg.display.set_caption("Mandelbrot Set")
        pg.key.set_repeat(210, 15)
        pg.display.flip()

        self.window = pg.display.set_mode((self.width, self.height), flags=pg.DOUBLEBUF|pg.SHOWN)


    def print_info(self):

        print("[{:>3}] fast zoom: {}, supersampling: {}x{}, performance: {}".format(
            self.level, self.fast_zoom, self.num_samples, self.num_samples, self.perf_level))
        print("[{:>3}] min-x, max-x : {:.16f}, {:.16f}".format(
            self.level, self.min_x, self.max_x))
        print("[{:>3}] min-y, max-y : {:.16f}, {:.16f}".format(
            self.level, self.min_y, self.max_y))
        print("[{:>3}] center x, y  : {:.16f}, {:.16f}".format(
            self.level, self.center_x, self.center_y))
        print("[{:>3}] zoom scale   : {}".format(
            self.level, str(self.zoom_scale)))
        print("[{:>3}] iterations   : {}".format(
            self.level, int(self.iters)))


    def run(self):

        self.display()

        # Wait for an event so minimum CPU utilization when idled.
        while True:
            e = pg.event.wait()
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_q or self.__on_key_press(e.key):
                    break
                pg.event.clear()
                sys.stdout.flush()
            elif e.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                self.__on_mouse_press(x, y, e.button)
                pg.event.clear()
            elif e.type == pg.VIDEOEXPOSE:
                self.update_window(videoexpose=True)
            elif e.type == pg.QUIT:
                break

        pg.quit()


    def update_iters(self, show_info=False):

        self.start_time = time.time()

        # Update min and max values as well.
        step_x = 4 / self.zoom_scale / self.width
        self.min_x = -2 / self.zoom_scale + self.center_x
        self.max_x = step_x * self.width + self.min_x - step_x

        step_y = -4 / self.zoom_scale / self.width
        self.min_y = 2 / self.zoom_scale * (self.height / self.width) + self.center_y
        self.min_y += 1e-15 if self.center_y == 0.0 else 0.0 # account for artifact
        self.max_y = step_y * self.height + self.min_y - step_y

        # Update iters which is a double value.
        self.iters = self.width / self.perf_win * 5.0 * \
            math.sqrt(math.pow(1/self.magf, self.level/math.pi))

        if show_info:
            self.print_info()

        return int(self.iters), step_x, step_y


    def update_level(self):

        if self.zoom_scale > 1.0:
            magn = 1.0
            while True:
                magn = 1 / self.magf * magn
                if magn > self.zoom_scale:
                    break
                self.level += 1
        elif self.zoom_scale < 1.0:
            magn = 1.0
            while True:
                magn = self.magf * magn
                if magn < self.zoom_scale:
                    break
                self.level -= 1


    def update_window(self, videoexpose=False):

        (temp, output) = (self.temp, self.output)
        buf = np.ravel(temp if self.num_samples == 1 else output)

        if not videoexpose:
            end_time = time.time() - self.start_time
            if self.rgb_sum_reset: self.rgb_sum = 0
            self.rgb_sum += self.__tally_rgb(buf.view(dtype=np.uint8))
            print("  RGB values total : {}".format(self.rgb_sum))
            print("      compute time : {:.3f} seconds".format(end_time))

        fmt = 'RGB' if len(temp.shape) == 3 and temp.shape[2] == 3 else 'RGBX'
        img = pg.image.frombuffer(buf, (self.width, self.height), fmt)

        self.window.blit(img, (0,0))
        pg.display.flip()


    def __handle_benchmark(self, symbol):

        self.rgb_sum = 0
        self.rgb_sum_reset = False

        if symbol == pg.K_z:
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 10042274467041.889
            self.center_x, self.center_y = -1.7485854655160802, 0.0126275661380999
        elif symbol == pg.K_x:
            # location courtesy of Tante Renate
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 1422616677257.3857
            self.center_x, self.center_y = -0.7746806106269039, -0.1374168856037867
        elif symbol == pg.K_c:
            # location courtesy of FracTest app
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 3068340407.9648294
            self.center_x, self.center_y = -0.0801706662109324, 0.6568767849265752
        elif symbol == pg.K_v:
            # location courtesy of Claude Heiland-Allen
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 29868296.757352185
            self.center_x, self.center_y = -1.7689786824401361, 0.0046053975000913
        elif symbol == pg.K_b:
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 3060043.2274685167
            self.center_x, self.center_y = -0.4391269761085219, 0.5745831678340557
        elif symbol == pg.K_g:
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 7273.144756668537
            self.center_x, self.center_y = -0.7426799200664, 0.16382294543399
        elif symbol == pg.K_t:
            self.level, self.zoom_scale, zoom_dest = 0, 0.33, 256749985.20257786
            self.center_x, self.center_y = -0.5476366631259864, 0.627355515204217
        else:
            return False

        self.update_level()
        self.display()
        time.sleep(0.45)
        pg.event.clear()

        start = time.time()

        if self.smooth_bench:
            interval = 1 / 24
            next_update = start + interval
            while self.zoom_scale < zoom_dest * 0.97:
                if pg.event.peek((pg.QUIT,), pump=True):
                    self.rgb_sum_reset = True
                    return True
                elif pg.event.peek((pg.KEYDOWN,), pump=True):
                    if pg.event.wait().key == pg.K_q:
                        return True
                    break
                self.__handle_zoom(1 / self.magf, 1)
                curr_time = time.time()
                if curr_time < next_update:
                    time.sleep(next_update - curr_time)
                next_update += interval
        else:
            while self.zoom_scale < zoom_dest * 0.97:
                if pg.event.peek((pg.QUIT,), pump=True):
                    self.rgb_sum_reset = True
                    return True
                elif pg.event.peek((pg.KEYDOWN,), pump=True):
                    if pg.event.wait().key == pg.K_q:
                        return True
                    break
                self.__handle_zoom(1 / self.magf, 1)

        print("    bench duration : {:.3f} seconds".format(time.time() - start))

        self.rgb_sum_reset = True
        return False


    def __handle_center(self, x, y):

        dx, dy = self.max_x - self.min_x, self.max_y - self.min_y

        self.center_x = self.min_x + (x * dx / self.width)
        self.center_y = self.min_y + (y * dy / self.height)

        self.display()


    def __handle_change_colors(self, symbol):

        palette_lkup = {
            pg.K_F1: 1, pg.K_F2: 2, pg.K_F3: 3, pg.K_F4: 4,
            pg.K_F5: 5, pg.K_F6: 6, pg.K_F7: 7,
        }

        self.color_scheme = palette_lkup[symbol]
        self.init_colors()

        print("[{:>3}] color scheme {}".format(self.level, self.color_scheme))

        self.display()


    def __handle_scroll_factor(self, symbol):

        if   symbol == pg.K_BACKQUOTE:     self.scroll_factor = 0.002
        elif symbol in (pg.K_1, pg.K_KP1): self.scroll_factor = 0.1
        elif symbol in (pg.K_2, pg.K_KP2): self.scroll_factor = 0.2
        elif symbol in (pg.K_3, pg.K_KP3): self.scroll_factor = 0.3
        elif symbol in (pg.K_4, pg.K_KP4): self.scroll_factor = 0.4
        elif symbol in (pg.K_5, pg.K_KP5): self.scroll_factor = 0.5
        elif symbol in (pg.K_6, pg.K_KP6): self.scroll_factor = 0.6
        elif symbol in (pg.K_7, pg.K_KP7): self.scroll_factor = 0.7
        elif symbol in (pg.K_8, pg.K_KP8): self.scroll_factor = 0.8
        elif symbol in (pg.K_9, pg.K_KP9): self.scroll_factor = 0.9
        elif symbol in (pg.K_0, pg.K_KP0): self.scroll_factor = 1.0

        print("[{:>3}] scroll factor {}".format(self.level, self.scroll_factor))


    def __handle_scroll_movement(self, symbol):

        if symbol in (pg.K_a, pg.K_LEFT):
            offset = abs(self.max_x - self.min_x) * self.scroll_factor
            self.center_x -= offset
        elif symbol in (pg.K_s, pg.K_DOWN):
            offset = abs(self.max_y - self.min_y) * self.scroll_factor
            self.center_y -= offset
        elif symbol in (pg.K_d, pg.K_RIGHT):
            offset = abs(self.max_x - self.min_x) * self.scroll_factor
            self.center_x += offset
        elif symbol in (pg.K_w, pg.K_UP):
            offset = abs(self.max_y - self.min_y) * self.scroll_factor
            self.center_y += offset

        self.display()


    def __handle_zoom(self, scale_factor, increment):

        self.zoom_scale = scale_factor * self.zoom_scale
        self.level += increment

        self.display()


    def __on_key_press(self, symbol):

        if symbol in (pg.K_LEFT, pg.K_DOWN, pg.K_RIGHT, pg.K_UP,
                      pg.K_a, pg.K_s, pg.K_d, pg.K_w):
            self.__handle_scroll_movement(symbol)

        elif symbol in (pg.K_BACKQUOTE,
                        pg.K_1, pg.K_6, pg.K_KP1, pg.K_KP6,
                        pg.K_2, pg.K_7, pg.K_KP2, pg.K_KP7,
                        pg.K_3, pg.K_8, pg.K_KP3, pg.K_KP8,
                        pg.K_4, pg.K_9, pg.K_KP4, pg.K_KP9,
                        pg.K_5, pg.K_0, pg.K_KP5, pg.K_KP0):
            self.__handle_scroll_factor(symbol)

        elif symbol in (pg.K_F1, pg.K_F2, pg.K_F3, pg.K_F4, pg.K_F5,
                        pg.K_F6, pg.K_F7):
            self.update_flag = True
            self.__handle_change_colors(symbol)

        elif symbol in (pg.K_z, pg.K_x, pg.K_c, pg.K_v, pg.K_b, pg.K_g, pg.K_t):
            return self.__handle_benchmark(symbol)

        elif symbol == pg.K_e:
            (temp, output) = (self.temp, self.output)
            buf = np.ravel(temp if self.num_samples == 1 else output)
            fmt = 'RGB' if len(temp.shape) == 3 and temp.shape[2] == 3 else 'RGBX'
            img = pg.image.frombuffer(buf, (self.width, self.height), fmt)
            pg.image.save(img, "image.png")
            print("Image saved as image.png.")

        elif symbol in (pg.K_i, pg.K_u, pg.K_h, pg.K_m, pg.K_n, pg.K_l):
            if   symbol == pg.K_i: self.num_samples = 9
            elif symbol == pg.K_u: self.num_samples = 7
            elif symbol == pg.K_h: self.num_samples = 5
            elif symbol == pg.K_m: self.num_samples = 3
            elif symbol == pg.K_n: self.num_samples = 2
            else:                  self.num_samples = 1

            self.update_flag = True
            self.init_offset()
            self.display()

        elif symbol in (pg.K_LEFTBRACKET, pg.K_PAGEDOWN):  # zoom in
            if 1 / self.magf * self.zoom_scale <= 2e13:
                self.__handle_zoom(1 / self.magf, 1)

        elif symbol in (pg.K_RIGHTBRACKET, pg.K_PAGEUP):  # zoom out
            if self.magf * self.zoom_scale >= 0.15:
                self.__handle_zoom(self.magf, -1)

        elif symbol in (pg.K_COMMA, pg.K_LESS):  # decrease performance
            if self.perf_level > 1:
                saveiters = int(self.iters)
                while True:
                    self.perf_level -= 1
                    self.perf_win = int(math.ceil(self.width / 800 * self.perf_level))
                    iters = self.width / self.perf_win * 5.0 * \
                        math.sqrt(math.pow(1/self.magf, self.level/math.pi))
                    if int(iters) != saveiters or self.perf_level == 1:
                        break

                self.display()

        elif symbol in (pg.K_PERIOD, pg.K_GREATER):  # increase performance
            if self.perf_level < 64 and self.iters >= 10:
                saveiters = int(self.iters)
                while True:
                    self.perf_level += 1
                    self.perf_win = int(math.ceil(self.width / 800 * self.perf_level))
                    iters = self.width / self.perf_win * 5.0 * \
                        math.sqrt(math.pow(1/self.magf, self.level/math.pi))
                    if int(iters) != saveiters:
                        break

                self.display()

        elif symbol in (pg.K_r, pg.K_HOME):  # reset display to initial view
            self.level, self.center_x, self.center_y, self.iters, self.zoom_scale = \
                self.reset_view

            self.display()

        return False


    def __on_mouse_press(self, x, y, button):

        if button == 1:
            if time.time() - self.start_click > 0.5:
                self.num_clicks = 1
            else:
                self.num_clicks += 1

            if self.num_clicks == 1:
                self.start_click = time.time()
            elif self.num_clicks >= 2:
                if self.num_clicks == 2 and time.time() - self.start_click <= 0.5:
                    self.__handle_center(x, y)
                self.num_clicks = 0


    def __tally_rgb(self, buf):

        t = np.sum(buf)

        # Subtract the alpha channel if present.
        if buf.shape[0] == self.height * self.width * 4:
            t -= 255 * self.height * self.width

        return int(t)

