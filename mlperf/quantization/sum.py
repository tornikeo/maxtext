import jax

import timing_util

_PROFILE=False
MATMUL_SIZES = [1024, 2048, 4096, 8192, 16384]



def f(X,Y):
  return X + Y

f_jit = jax.jit(f)

for matrix_size in MATMUL_SIZES:
  A = jax.numpy.ones ( (matrix_size, matrix_size), dtype=jax.numpy.bfloat16)
  B = jax.numpy.ones ( ( matrix_size, matrix_size), dtype=jax.numpy.bfloat16)
  print(f'size of A  is {A.size} and dim {matrix_size}')
  num_bytes_per_entry = 2 # blfloat16
  num_matrices_stored = 3 #(A, B, A+B)
  num_gb_bytes = (num_matrices_stored * num_bytes_per_entry * A.size)/10**9
  num_tflops = (matrix_size * matrix_size * 2)/10**12 # add and mult
  average_time_sec = timing_util.simple_timeit(f, A, B, task = 'sum_' + str(matrix_size), enable_profile=_PROFILE)
  print(f'Achieved tflops = {num_tflops/average_time_sec} and achieved bw gbps = {num_gb_bytes/average_time_sec}')
  print(f'Comp time s = {num_tflops/197} and Comm time s = {num_gb_bytes/817}')

  jit_average_time_sec = timing_util.simple_timeit(f_jit, A, B, task = 'jit_sum_' + str(matrix_size), enable_profile=_PROFILE)
  print(f'Jit Achieved tflops = {num_tflops/jit_average_time_sec} and achieved bw gbps = {num_gb_bytes/jit_average_time_sec}')
  print(f'Jit Comp time s = {num_tflops/197} and Comm time s = {num_gb_bytes/817}')




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
