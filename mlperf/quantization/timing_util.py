import datetime
import jax
import random
import string


def simple_timeit(f, *args, tries = 10, task = None, enable_profile=False):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"{task}" # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"/tmp/{trace_name}"
    print(trace_dir)

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    if enable_profile:
      jax.profiler.start_trace(trace_dir)
    for _ in range(tries):
      s = datetime.datetime.now()
      jax.block_until_ready(f(*args))
      e = datetime.datetime.now()
      outcomes_ms.append(1000*(e-s).total_seconds())
    if enable_profile:
      jax.profiler.stop_trace()
    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    print(f'Average time ms for mm for {task} is {round(average_time_ms, 3)}')
    return average_time_ms/1000


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

