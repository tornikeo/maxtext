import jax

import timing_util

_PROFILE=False
MATMUL_SIZES = [(250, 2048)]

_INT4 = jax.numpy.int4
_INT8 = jax.numpy.int8
_DEFAULT = jax.numpy.bfloat16


def f(X,Y):
  return jax.lax.batch_matmul(X,Y)

f_jit = jax.jit(f)

num_matmuls, matrix_size = MATMUL_SIZES[0]

for (dtypeA, dtypeB) in [(_INT4, _INT4), (_INT4, _INT8), (_INT8, _INT4), (_INT8, _INT8), (_INT8, _DEFAULT), (_DEFAULT, _DEFAULT)]:
  A = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=dtypeA)
  B = jax.numpy.ones ( (num_matmuls, matrix_size, matrix_size), dtype=dtypeB)

  print(f'A, B shape is {f(A, B).shape}. A dtype is {A.dtype}, B dtype is {B.dtype} and prod type is {f(A, B).dtype}')
  timing_util.simple_timeit(f_jit, A, B, task = 'matmul_' + str(matrix_size), enable_profile=_PROFILE)


# Run this script
# Run tensorboard --logdir /tmp/<dirname> to see profile in tensorboard.


##  Results on v5e - Not much gains from going to int8 from int4 but significant benefit in ging from bfloat16 to int8.
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
# A, B shape is (250, 2048, 2048). A dtype is int4, B dtype is int4 and prod type is int4
# Average time ms for mm for matmul_2048 is 12.4075
# A, B shape is (250, 2048, 2048). A dtype is int4, B dtype is int8 and prod type is int8
# Average time ms for mm for matmul_2048 is 11.376100000000001
# A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is int4 and prod type is int8
# Average time ms for mm for matmul_2048 is 12.1586
# A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is int8 and prod type is int8
# Average time ms for mm for matmul_2048 is 12.109599999999997
# A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_2048 is 23.292099999999998
# A, B shape is (250, 2048, 2048). A dtype is bfloat16, B dtype is bfloat16 and prod type is bfloat16
# Average time ms for mm for matmul_2048 is 24.4792


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

'''
v6e results
(.env) msingh@t1v-n-4ed3b76a-w-0:~/maxtext/mlperf/quantization$ python3  matmul.py
A, B shape is (64000, 128, 128) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_128
Average time ms for mm for matmul_128 is 5.008
A, B shape is (16000, 256, 256) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_256
Average time ms for mm for matmul_256 is 5.002
A, B shape is (4000, 512, 512) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_512
Average time ms for mm for matmul_512 is 5.025
A, B shape is (3000, 640, 640) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_640
Average time ms for mm for matmul_640 is 5.909
A, B shape is (2000, 768, 768) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_768
Average time ms for mm for matmul_768 is 5.649
A, B shape is (1000, 1024, 1024) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_1024
Average time ms for mm for matmul_1024 is 5.75
A, B shape is (250, 2048, 2048) and A, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 8.033
(.env) msingh@t1v-n-4ed3b76a-w-0:~/maxtext/mlperf/quantization$ ls
__pycache__  matmul.py  matmul_dtype.py  sum.py  timing_util.py
(.env) msingh@t1v-n-4ed3b76a-w-0:~/maxtext/mlperf/quantization$ python3  matmul_dtype.py
A, B shape is (250, 2048, 2048). A dtype is int4, B dtype is int4 and prod type is int4
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 1.975
A, B shape is (250, 2048, 2048). A dtype is int4, B dtype is int8 and prod type is int8
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 3.257
A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is int4 and prod type is int8
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 2.994
A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is int8 and prod type is int8
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 3.442
A, B shape is (250, 2048, 2048). A dtype is int8, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 7.481
A, B shape is (250, 2048, 2048). A dtype is bfloat16, B dtype is bfloat16 and prod type is bfloat16
/tmp/matmul_2048
Average time ms for mm for matmul_2048 is 8.026
'''
