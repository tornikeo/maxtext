import jax

import timing_util

_PROFILE=False
MATMUL_SIZES = [(64000,128), (16000,256), (4000,512), (3000,640), (2000,768), (1000,1024), (250, 2048)]

_INT4 = jax.numpy.int4
_INT8 = jax.numpy.int8
_DEFAULT = jax.numpy.bfloat16


def f(X,Y):
  return jax.lax.batch_matmul(X,Y)

f_jit = jax.jit(f)

for num_matmuls, matrix_size in MATMUL_SIZES:
  A = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=jax.numpy.bfloat16)
  B = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=jax.numpy.bfloat16)

  print(f'A, B shape is {f(A, B).shape} and A, B dtype is {A.dtype} and prod type is {f(A, B).dtype}')
  timing_util.simple_timeit(f_jit, A, B, task = 'matmul_' + str(matrix_size), enable_profile=_PROFILE)


# Run this script
# Run tensorboard --logdir /tmp/<dirname> to see profile in tensorboard.


# A, B shape is (64000, 128, 128) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_128 is 9.3185
# A, B shape is (16000, 256, 256) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_256 is 9.3416
# A, B shape is (4000, 512, 512) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_512 is 11.325999999999999
# A, B shape is (3000, 640, 640) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_640 is 13.230799999999999
# A, B shape is (2000, 768, 768) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_768 is 11.2064
# A, B shape is (1000, 1024, 1024) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_1024 is 12.908399999999997
# A, B shape is (250, 2048, 2048) and A, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_2048 is 24.5258



# # https://docs.google.com/spreadsheets/d/1tDLT25HzX65m7tIH80LW3LB2-XhlbDRVjGlzfN5m64w/edit#gid=1538397185
# v5e_per_chip = {
#     'tflops_peak_bf16': 197,
#     'tflops_peak_int8': 393,
#     'hbm_gb': 16,
#     'hbm_bw_gb_per_sec': 817,
#     'ici_bw_gbps': 1600,
#     'torus': 2,
#     'price-per-chip-hr-1yr': 1.2,
#     'price-per-chip-hr-3yr': 0.54
# }

# Each v4 chip is 275 TFLOPs HBM (32 GB, 1200 GB/s) and ICI BW is (2400)

# Sum Example:
# Bytes = Size(A) + Size(B) + Size(A+B) *2 for bfloat16 => 6 *matrix_dim * matrix_dim
# Flops = matrix_dim * matrix_dim


# MM Example
