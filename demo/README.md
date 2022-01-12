# Mandelbrot Set

Non-parallel demonstrations for generating the Mandelbrot Set.

I begin with a cool example from the web and make 4 examples to showcase Numba
running on a single core. The examples demonstrate Anti-Aliasing, Gaussian Blur,
and Unsharp Mask via a step-by-step approach.

<p align="center">
  <img src="../../assets/demo.png?raw=true" alt="Mandelbrot Set"/>
</p>

**Python, no JIT**

```bash
$ python3 ex0.py 
     mandelbrot 5.374 seconds
  image saved as img1.png
```

**Python, JIT**

```bash
$ python3 ex1.py 
       jit time 0.340 seconds
     mandelbrot 0.087 seconds
  image saved as img1.png

$ python3 ex2.py 
       jit time 0.948 seconds
     mandelbrot 0.087 seconds
  anti-aliasing 0.424 seconds
  image saved as img2.png

$ python3 ex3.py 
       jit time 1.152 seconds
     mandelbrot 0.087 seconds
  anti-aliasing 0.424 seconds
  gaussian blur 0.013 seconds
  image saved as img3.png

$ python3 ex4.py 
       jit time 1.313 seconds
     mandelbrot 0.087 seconds
  anti-aliasing 0.428 seconds
  gaussian blur 0.013 seconds
   unsharp mask 0.001 seconds
  image saved as img4.png
```

# Acknowledgements

A cool demonstration on the web that inspired me to add a demo folder.

* [Visualizing the Mandelbrot Set Using Python by Blake Sanie](https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f)

I learned to apply Anti-Aliasing from a Mandelbrot demonstration by @josch.

* [Calculating Mandelbrot using different ways](https://github.com/josch/mandelbrot)

Unsharp Mask was adapted from Pillow code.

* [Pillow UnsharpMask](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/UnsharpMask.c)

