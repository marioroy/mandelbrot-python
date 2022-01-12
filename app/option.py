# -*- coding: utf-8 -*-
"""
Provides the Option class for config file and command line parsing.
This pre-populates OPT during import with defaults and parsed options.
"""

__all__ = ['OPT']

import os, sys

from configparser import ConfigParser
from optparse import OptionGroup, OptionParser
from os.path import basename, exists

class Option(object):

    def __init__(self):

        usage = "%prog [--config filepath [section]] [options]"
        epilog = """
          Values exceeding the range specification are silently clipped to
          the respective minimum or maximum value. The number of iterations
          is computed dynamically based on the performance level
          (lower equals more iterations).
          """
        epilog = " ".join([line.lstrip() for line in epilog.splitlines()])

        p = OptionParser(usage=usage, version="%prog 0.1.0", epilog=epilog)

        def _opt(parser, opt, t, h):
          if t is None:
            parser.add_option(opt, help=h, action="store_true", default=False)
          else:
            parser.add_option(opt, type=t, help=h, metavar="ARG")

        # allow options with underscore by replacing with dash
        for i in range(1, len(sys.argv)):
            if sys.argv[i].startswith('--'):
                sys.argv[i] = sys.argv[i].replace('_', '-')

        # configure options
        _opt(p, "--shortcuts", None, "show keyboard shortcuts and exit")
        _opt(p, "--width", "int", "width of window [100-8000]: 800")
        _opt(p, "--height", "int", "height of window [100-5000]: 500")
        _opt(p, "--location", "int", "begin at app location [0-7]: 0")
        _opt(p, "--center-x", "float", "home center-x value [float]: -0.625")
        _opt(p, "--center-y", "float", "home center-y value [float]: 0.0")
        _opt(p, "--zoom-scale", "float", "home zoom magnification [float]: 0.95")
        _opt(p, "--num-samples", "int", "anti-aliasing factor [1-9]: 2")
        _opt(p, "--perf-level", "int", "performance level [1-64]: 25")
        _opt(p, "--color-scheme", "int", "select color scheme [1-7]: 1")
        _opt(p, "--fast-zoom", "int", "select fast zoom [0,1]: 1")
        _opt(p, "--smooth-bench", "int", "select smooth bench [0,1]: 0")

        g = OptionGroup(p, "CPU Options (mandel_parfor, mandel_queue, mandel_stream)")
        _opt(g, "--num-threads", "string", "number of threads to use: auto")
        p.add_option_group(g)

        g = OptionGroup(p, "CUDA Options (mandel_cuda)")
        _opt(g, "--compiler-bindir", "str", "directory in which the host C compiler resides")
        p.add_option_group(g)

        g = OptionGroup(p, "GPU Options (mandel_cuda, mandel_ocl)")
        _opt(g, "--mixed-prec", "int", "select mixed-precision flag [0,1,2]: 0")
        _opt(g, "--fma", "int", "select fused-multiply-add flag [0,1]: 0")
        p.add_option_group(g)

        p.set_defaults(
            width=800, height=500, center_x=-0.625, center_y=0.0, location=0,
            zoom_scale=0.95, num_samples=2, perf_level=25, color_scheme=1,
            fast_zoom=1, smooth_bench=0, num_threads='auto', compiler_bindir='',
            mixed_prec=0, fma=0 )

        # optionally, override defaults from a config file
        self.__handle_config(p)

        # process command-line arguments
        (opt, args) = p.parse_args()

        # show usage
        if len(args):
            p.print_help()
            sys.exit(2)
        if opt.shortcuts:
            show_keyboard_shortcuts()
            sys.exit(0)

        # clamp to minimum-maximum values
        self.width = max(100, min(8000, opt.width))
        self.height = max(100, min(5000, opt.height))
        self.location = max(0, min(7, opt.location))
        self.num_samples = max(1, min(9, opt.num_samples))
        self.perf_level = max(1, min(64, opt.perf_level))
        self.color_scheme = max(1, min(7, opt.color_scheme))
        self.fast_zoom = max(0, min(1, opt.fast_zoom))
        self.smooth_bench = max(0, min(1, opt.smooth_bench))
        self.mixed_prec = max(0, min(2, opt.mixed_prec))
        self.fma = max(0, min(1, opt.fma))
        self.center_x = opt.center_x
        self.center_y = opt.center_y
        self.zoom_scale = opt.zoom_scale
        self.compiler_bindir = opt.compiler_bindir

        if opt.num_threads != 'auto':
            self.num_threads = max(1, int(opt.num_threads))
        else:
            ncpu = int(
                os.getenv('NUMBA_NUM_THREADS') or
                os.getenv('NUM_THREADS') or
                os.cpu_count() - 1
                )
            self.num_threads = max(1, ncpu)

        del opt, args


    @classmethod
    def __handle_config(cls, parser):

        if len(sys.argv) >= 2 and sys.argv[1].startswith('--config'):
            try:
                (_, config_path) = sys.argv[1].split('=')
                del sys.argv[1]
            except ValueError:
                config_path = sys.argv[2]
                del sys.argv[2], sys.argv[1]

            if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
                section = sys.argv[1]
                del sys.argv[1]
            else:
                section = 'common'

            if not exists(config_path):
                prog = basename(sys.argv[0])
                mesg = f"{prog}: error: no such file or directory: '{config_path}'"
                print(mesg, file=sys.stderr)
                sys.exit(2)

            config = ConfigParser(default_section=None, empty_lines_in_values=False)
            config.read(config_path)

            cls.__override_defaults(parser, config, 'common')
            if section != 'common':
                cls.__override_defaults(parser, config, section)


    @classmethod
    def __override_defaults(cls, parser, config, section):

        if not config.has_section(section):
            prog = basename(sys.argv[0])
            mesg = f"{prog}: error: no such section in config: '{section}'"
            print(mesg, file=sys.stderr)
            sys.exit(2)

        opt = dict()

        for key in ('width', 'height', 'location', 'num_samples', 'perf_level',
                    'color_scheme', 'fast_zoom', 'smooth_bench', 'mixed_prec',
                    'fma'):
            if config.has_option(section, key):
                opt[key] = int(config.get(section, key))

        for key in ('center_x', 'center_y', 'zoom_scale'):
            if config.has_option(section, key):
                opt[key] = float(config.get(section, key))

        for key in ('num_threads', 'compiler_bindir'):
            if config.has_option(section, key):
                opt[key] = str(config.get(section, key))

        if len(opt):
            parser.set_defaults(**opt)


def show_keyboard_shortcuts():

    print("""
Keyboard shortcuts:
  Most all options are accessible via a keyboard shortcut.
  Zooming does not exceed double-precision limit.
  q)         terminate the application and exit
  r) Home)   reset window back to the home location
  [) PageDn) zoom in from the center of the window
  ]) PageUp) zoom out from the center of the window
  e)         export the window RGB values to image.png
  a) Left)   scroll window left
  s) Down)   scroll window down
  d) Right)  scroll window right
  w) Up)     scroll window up

  Auto zoom:
    Specify option --fast-zoom for fast/slow zoom mode.
    Specify option --smooth-bench for smooth/bench mode.
    Press any key to end auto zoom.
    z)       auto zoom from scale 0.33 to near --location 1
    x)       auto zoom from scale 0.33 to near --location 2
    c)       auto zoom from scale 0.33 to near --location 3
    v)       auto zoom from scale 0.33 to near --location 4
    b)       auto zoom from scale 0.33 to near --location 5
    g)       auto zoom from scale 0.33 to near --location 6
    t)       auto zoom from scale 0.33 to near --location 7

  Color scheme:
    F1-F7)   select color scheme 1 through 7, respectively

  Image quality:
    Supersampling involves sharpening using Unsharp Mask.
    l)ow     render without supersampling
    n)       render with 2x2 supersampling
    m)edium  render with 3x3 supersampling
    h)igh    render with 5x5 supersampling
    u)ltra   render with 7x7 supersampling
    i)       render with 9x9 supersampling

  Performance level:
    Iterations are computed dynamically based on the performance level.
    <) ,)    decreasing performance level increases iterations ;min 1
    >) .)    increasing performance level decreases iterations ;max 64

  Scroll movement:
    Double clicking sets center-x,center-y to the mouse x,y coordinates
    and renders the image.
    `)       set scroll factor to 0.002x for micro adjustments
    1)       set scroll factor to 0.1x width or height
    2)       set scroll factor to 0.2x width or height
    3)       set scroll factor to 0.3x width or height
    4)       set scroll factor to 0.4x width or height
    5)       set scroll factor to 0.5x width or height
    6)       set scroll factor to 0.6x width or height
    7)       set scroll factor to 0.7x width or height
    8)       set scroll factor to 0.8x width or height
    9)       set scroll factor to 0.9x width or height
    0)       set scroll factor to 1.0x width or height
    """.strip())


# Instantiate so that OPT is pre-populated with options during import.

if __name__ == '__main__':
    OPT = Option()
    print(vars(OPT))
else:
    OPT = Option()

