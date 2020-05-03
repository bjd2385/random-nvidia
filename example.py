#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
First attempt at pycuda -- example taken from

https://documen.tician.de/pycuda/

the home page :P
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
from time import time


# Kernel to multiply 2 400-element vectors :D
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  int ai = a[i];
  int bi = b[i];
  dest[i] = (ai + bi);
  int j;
  for (j = 0; j < 10000; j++) dest[i] *= (1 / (ai + bi));
}
""")


# Make a function from the kernel.
multiply_them = mod.get_function("multiply_them")

# Allocate vectors in CPU / MOBO memory.
a = numpy.random.randn(40000000).astype(numpy.float32)
b = numpy.random.randn(40000000).astype(numpy.float32)

# Destination array.
dest = numpy.zeros_like(a)

start = time()

# Now perform the multiplication, from what's available above.
multiply_them(
    drv.Out(dest),  # Destination
    drv.In(a),      # First array to work with
    drv.In(b),      # Second array to work with
    block=(4, 1, 1), grid=(1, 1)    #
)


print(f'Time: {time() - start}')