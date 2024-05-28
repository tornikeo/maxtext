# """
# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example:
# python MaxText/inference-microbenchmark-dataset.py MaxText/configs/base.yml async_checkpointing=false attention=autoselected \
#  model_name=llama2-7b weight_dtype=bfloat16 tokenizer_path=assets/tokenizer.llama2 scan_layers=false run_name=test_mb\
#  dataset_path=gs://maxtext-dataset ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 async_checkpointing=false \
#  base_output_directory=gs://${USER}-maxtext-outputs max_prefill_predict_length=1024 max_target_length=2048 per_device_batch_size=2 quantization="" quantize_kvcache=False  \
#  load_parameters_path=gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items
#
# Enable profiler by adding arg --enable_profiler=True
# """



"""Inference microbenchmark for prefill and autoregressive steps."""
import datetime
import jax
import json
import statistics
import sys

from jetstream.engine import token_utils

import max_utils
import maxengine
import maxtext_utils
import pyconfig
import os

_WARMUP_ITERS = 2 # Number of batches in dataset used for warmup
_PREFILL_LENGTH = 1024 # For padding input tokens

_NUM_SAMPLES=100
_DATAFILE = f"/home/msingh/datasets/openorca_json/output_{_NUM_SAMPLES}.json"
_OUTFILE=f"/home/msingh/benchmarks/openorca_json/rouge_{_NUM_SAMPLES}.json"

_PREFILL = "prefill"
_PREFILL_INSERT = "prefill_insert"
_AUTOREGRESSIVE = "autoregressive"

_MSEC_PER_TOK = 'msec_per_token'
_MSEC_PER_SEQ = 'msec_per_sequence'


def read_openorca_dataset(filepath, print_stats=True):
  assert os.path.exists(filepath), f'input dataset file: {filepath}  does not exist'
  with open(filepath) as f:
    dataset = json.load(f)

  if print_stats:
    n = len(dataset)
    len_prompt_tokens = []
    len_output_tokens = []
    len_total_tokens = []
    for d in dataset:
      len_prompt_tokens.append(d['len_prompt_tokens'])
      len_output_tokens.append(d['len_output_tokens'])
      len_total_tokens.append(d['len_prompt_tokens'] + d['len_output_tokens'])
    print(f'\nStats for dataset in file: {filepath}')
    print(f'Num requests: {n}')
    max_len = max(len_prompt_tokens)
    min_len = min(len_prompt_tokens)
    quantiles = statistics.quantiles(len_prompt_tokens, n=10)
    print(f'Distribution len prompt tokens: max: {max_len}, min: {min_len}, deciles: {quantiles}')
    max_len = max(len_output_tokens)
    min_len = min(len_output_tokens)
    quantiles = statistics.quantiles(len_output_tokens, n=10)
    print(f'Distribution len output tokens: max: {max_len}, min {min_len}, deciles: {quantiles}')
    max_len = max(len_total_tokens)
    min_len = min(len_total_tokens)
    quantiles = statistics.quantiles(len_total_tokens, n=10)
    print(f'Distribution len total tokens: max: {max_len}, min {min_len}, deciles: {quantiles}')
    print('\n')
  return dataset

def tokenize_dataset(engine, dataset, prefill_length):
  tokenized_dataset = []
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  for i in range(len(dataset)):
    datarow = dataset[i]
    tokens, true_length = token_utils.tokenize_and_pad(
        datarow['prompt'], vocab, is_bos=True, prefill_lengths=[prefill_length])
    # Diff of 1 token because of bos prepended to list of tokens.
    prompt_token_len =  datarow['len_prompt_tokens'] + 1
    assert true_length == prompt_token_len, f'{true_length=} mismatches {prompt_token_len=}'
    tokenized_dataset.append((tokens, true_length))
  return tokenized_dataset


def benchmark_prefill(config, engine, params, tokenized_data, warmup):
  """Benchmarking prefill step."""
  if not warmup:
    max_utils.activate_profiler(config, _PREFILL)
  start = datetime.datetime.now()
  for (tokens, true_length) in tokenized_data:
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  time_seconds = (end - start).total_seconds()
  if not warmup:
    max_utils.deactivate_profiler(config)
  max_utils.delete_pytree(prefill_result)
  stats = {
    _MSEC_PER_SEQ: round(time_seconds*1000/len(tokenized_data), 2)
  }
  return stats

def benchmark_prefill_insert(config, engine, params, decode_state, tokenized_data, warmup):
  """Benchmarking prefill and insert step."""
  if not warmup:
    max_utils.activate_profiler(config, _PREFILL_INSERT)
  start = datetime.datetime.now()
  total_slots = engine.max_concurrent_decodes
  for i in range(len(tokenized_data)):
    (tokens, true_length) = tokenized_data[i]
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, decode_state, int(i % total_slots))
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  time_seconds = (end - start).total_seconds()
  if not warmup:
    max_utils.deactivate_profiler(config)
  max_utils.delete_pytree(prefill_result)
  stats = {
    _MSEC_PER_SEQ: round(time_seconds*1000/len(tokenized_data), 2)
  }
  return stats, decode_state

def benchmark_autoregressive(config, engine, params, decode_state, batch_size, warmup):
  """Benchmarking autoregressive step."""
  sampled_tokens_list = []
  steps = range(config.max_prefill_predict_length, config.max_target_length)
  if not warmup:
    max_utils.activate_profiler(config, _AUTOREGRESSIVE)
  start = datetime.datetime.now()
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(sampled_tokens)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  if not warmup:
    max_utils.deactivate_profiler(config)
  time_seconds = (end - start).total_seconds()
  stats = {
    _MSEC_PER_TOK: round(time_seconds*1000/(steps*batch_size), 2)
  }
  return stats, sampled_tokens_list

def benchmark(config, engine, params, num_params, decode_state, tokenized_batch, warmup):
    batch_size = len(tokenized_batch)
    prefill_stats = benchmark_prefill(config, engine, params, tokenized_batch, warmup)
    prefill_insert_stats = None
    #prefill_insert_stats, decode_state = benchmark_prefill_insert(config, engine, params, decode_state, tokenized_batch, warmup)
    ar_stats = None
    output_tokens = None
    #ar_stats, output_tokens = benchmark_autoregressive(config, engine, params, decode_state, batch_size, warmup)
    batch_stats = (prefill_stats, prefill_insert_stats, ar_stats)
    return batch_stats, output_tokens

def get_quantiles(stats_array):
  return f'min: {min(stats_array)}, max: {max(stats_array)}, deciles:{ round(s, 2) for s in statistics.quantiles(stats_array, n=10)}'

def aggregate_stats(stats_list):
  output_stats = {}
  output_stats[f'{_PREFILL}-{_MSEC_PER_SEQ}'] = []
  output_stats[f'{_PREFILL_INSERT}-{_MSEC_PER_SEQ}'] = []
  output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_TOK}'] = []
  for (prefill_stats, prefill_insert_stats, ar_stats) in stats_list:
    if prefill_stats and _MSEC_PER_SEQ in prefill_stats:
      output_stats[f'{_PREFILL}-{_MSEC_PER_SEQ}'].append(prefill_stats[_MSEC_PER_SEQ])
    if prefill_insert_stats and _MSEC_PER_SEQ in prefill_insert_stats:
      output_stats[f'{_PREFILL_INSERT}-{_MSEC_PER_SEQ}'].append(prefill_insert_stats[_MSEC_PER_SEQ])
    if ar_stats and _MSEC_PER_TOK in ar_stats:
      output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_TOK}'].append(ar_stats[_MSEC_PER_TOK])

  result = {}
  for k in output_stats.keys():
    if len(output_stats[k]) > 0:
      result[k] = get_quantiles(output_stats[k])
  return result


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  # Set up initial state
  decode_state = engine.init_decode_state()
  # cache_shape = jax.tree.map(lambda x: x.shape, decode_state["cache"])
  # print(f"Cache shape {cache_shape}")
  _, cache_size, _ = max_utils.summarize_pytree_data(decode_state["cache"], name="Cache")
  num_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")
  batch_size = engine.max_concurrent_decodes

  # Read the dataset
  dataset = read_openorca_dataset(_DATAFILE)
  dataset_size = len(dataset)
  tokenized_dataset = tokenize_dataset(engine, dataset, _PREFILL_LENGTH)
  assert len(tokenized_dataset) == dataset_size

  # Warmup
  print("Warmup started ..")
  for _ in range(_WARMUP_ITERS):
    e = min(batch_size, dataset_size)
    tokenized_batch = tokenized_dataset[0:e]
    # print(f'Warmup - batch (0:{e})')
    # print([f'num_tokens:{len(t)}, true_len:{l}' for (t,l) in tokenized_batch])
    # print()
    _, _= benchmark(config, engine, params, num_params, decode_state, tokenized_batch, True)

  # Benchmark
  stats = []
  print(f"\nBenchmarks started: dataset_size {dataset_size} and batch_size {batch_size}")
  for s in range(0, dataset_size, batch_size):
    e = min(s + batch_size, dataset_size)
    tokenized_batch = tokenized_dataset[s:e]
    # print(f'Benchmark - iter:({s}:{e})')
    # print([f'num_tokens:{len(t)}, true_len:{l}' for (t,l) in tokenized_batch])
    batch_stats, batch_output_tokens = benchmark(config, engine, params, num_params, decode_state, tokenized_batch, False)
    # print(f" batch ({s}:{e}), batch_stats {batch_stats}")
    stats.append(batch_stats)

  # Aggregate stats
  print('\nAggregating stats')
  result = aggregate_stats(stats)

  # Output stats
  print('\nResults')
  for k in result.keys():
    print('\t' + f'{k}: {result[k]}')


  print('\nEnd')
  # TBD - add rouge score

if __name__ == "__main__":
  pyconfig.initialize(sys.argv)
  main(pyconfig.config)
