gcc dither_gpu.c -o dither_gpu -std=c11 -O2 -lpng -lOpenCL -lm
gcc dither_cpu.c -o dither_cpu -O2 -lpng -lm