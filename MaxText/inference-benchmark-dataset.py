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
# Enable profiler by adding arg: enable_profiler=True
# """

# pip install nltk
# pip install rouge_score
# pip install evaluate

"""Inference microbenchmark for prefill and autoregressive steps."""
import datetime
import evaluate
import jax
import json
import numpy as np
import statistics
import sys

from jetstream.engine import token_utils

import max_utils
import maxengine
import maxtext_utils
import nltk

import pyconfig
import os

_WARMUP_ITERS = 1 # Number of batches in dataset used for warmup
_PREFILL_LENGTH = 1024 # For padding input tokens

_NUM_SAMPLES=24000
#_NUM_SAMPLES=10
_DATAFILE = f"/home/msingh/datasets/openorca_json/output_{_NUM_SAMPLES}.json"

_PREFILL = "prefill"
_PREFILL_INSERT = "prefill_insert"
_AUTOREGRESSIVE = "autoregressive"

_MSEC_PER_SEQ_PER_TOK = 'msec_per_seq_per_token'
_MSEC_PER_TOK = 'msec_per_token'
_MSEC_PER_SEQ = 'msec_per_sequence'
_TOKENS_PER_SEC = 'tokens/sec'

_PRINT_BATCH_STATS = False
def get_quantiles(name, stats_array):
  return f'{name} -  min: {min(stats_array)}, max: {max(stats_array)}, deciles:{[round(s, 2) for s in statistics.quantiles(stats_array, n=10)]}'

def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]
  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]
  return preds, targets


def eval_accuracy(predicted, target):
  assert len(predicted) == len(target), 'Mismatched predicted and target outputs.'
  metric = evaluate.load("rouge")
  nltk.download("punkt")
  preds = []
  targets = []

  preds, targets = postprocess_text(predicted, target)
  result = metric.compute(
      predictions=preds,
      references=targets,
      use_stemmer=True,
      use_aggregator=False,
  )
  result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  target_lens = [len(tgt) for tgt in targets]
  result["num_requests"] = len(preds)
  result["sum_len_preds"] = int(np.sum(prediction_lens))
  result["sum_len_targets"] = int(np.sum(target_lens))
  result["num_requests"] = len(preds)
  return result


def read_openorca_dataset(filepath, print_stats=True):
  assert os.path.exists(filepath), f'input dataset file: {filepath}  does not exist'
  # Read file
  with open(filepath) as f:
    dataset = json.load(f)
  # Print stats
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
    print(get_quantiles('len_prompt_tokens', len_prompt_tokens))
    print(get_quantiles('len_output_tokens', len_output_tokens))
    print(get_quantiles('len_total_tokens', len_total_tokens))
    print('\n')
  return dataset


def benchmark_prefill(config, engine, params, tokens, true_lengths, start_row):
  """Benchmarking prefill step."""
  batch_size = len(tokens)
  # Warmup
  if start_row == 0:
    for i in range(_WARMUP_ITERS):
      prefill_result = engine.prefill(params=params, padded_tokens=tokens[i], true_length=true_lengths[i])
    jax.block_until_ready(prefill_result)
    max_utils.delete_pytree(prefill_result)
  # Benchmark
  print("benchmark prefill")
  max_utils.activate_profiler(config, f'{_PREFILL}_{start_row}')
  start = datetime.datetime.now()
  for i in range(batch_size):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens[i], true_length=true_lengths[i])
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  time_seconds = (end - start).total_seconds()
  max_utils.deactivate_profiler(config)
  max_utils.delete_pytree(prefill_result)
  # Stats
  stats = {
    _MSEC_PER_SEQ: round(time_seconds*1000/batch_size, 2)
  }
  return stats

def benchmark_prefill_insert(config, engine, decode_state, params, tokens, true_lengths, start_row):
  """Benchmarking prefill and insert step."""
  batch_size = len(tokens)
  total_slots = engine.max_concurrent_decodes
  # Warmup
  if start_row == 0:
    for i in range(_WARMUP_ITERS):
      prefill_result = engine.prefill(params=params, padded_tokens=tokens[i], true_length=true_lengths[i])
      decode_state = engine.insert(prefill_result, decode_state, int(i % total_slots))
    jax.block_until_ready(decode_state)
    max_utils.delete_pytree(prefill_result)
  # Benchmark
  print("benchmark prefill insert")
  max_utils.activate_profiler(config, f'{_PREFILL_INSERT}_{start_row}:{start_row+batch_size}')
  start = datetime.datetime.now()
  for i in range(batch_size):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens[i], true_length=true_lengths[i])
    decode_state = engine.insert(prefill_result, decode_state, int(i % total_slots))
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  time_seconds = (end - start).total_seconds()
  max_utils.deactivate_profiler(config)
  max_utils.delete_pytree(prefill_result)

  stats = {
    _MSEC_PER_SEQ: round(time_seconds*1000/batch_size, 2)
  }
  return stats, decode_state

def benchmark_autoregressive(config, engine, decode_state, params, batch_size, start_row):
  """Benchmarking autoregressive step."""
  steps = range(config.max_prefill_predict_length, config.max_target_length)
  # Warmup
  if start_row == 0:
    for i in range(_WARMUP_ITERS):
      decode_state, sampled_tokens = engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)
  # Benchmark
  print("benchmark autoregressive")
  sampled_tokens_list = []
  max_utils.activate_profiler(config, f'{_AUTOREGRESSIVE}_{start_row}:{start_row+batch_size}')
  start = datetime.datetime.now()
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(sampled_tokens)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  max_utils.deactivate_profiler(config)
  max_utils.delete_pytree(decode_state)
  time_seconds = (end - start).total_seconds()
  # Stats
  stats = {
    _MSEC_PER_SEQ_PER_TOK: round(time_seconds*1000/(len(steps)), 2),
    _MSEC_PER_TOK: round(time_seconds*1000/(len(steps)*batch_size), 2),
    _TOKENS_PER_SEC: round(batch_size*len(steps)/time_seconds, 2)
  }
  return stats, sampled_tokens_list

def benchmark(config, engine, decode_state, params, tokens, true_lengths, suffix):
    batch_size = len(tokens)
    prefill_stats = None
    prefill_stats = benchmark_prefill(config, engine, params, tokens, true_lengths, suffix)
    prefill_insert_stats = None
    prefill_insert_stats, decode_state = benchmark_prefill_insert(config, engine, decode_state, params, tokens, true_lengths, suffix)
    ar_stats = None
    output_tokens = None
    ar_stats, output_tokens = benchmark_autoregressive(config, engine, decode_state, params, batch_size, suffix)
    batch_stats = (prefill_stats, prefill_insert_stats, ar_stats)
    return batch_stats, output_tokens


def aggregate_stats(stats_list):
  output_stats = {}
  output_stats[f'{_PREFILL}-{_MSEC_PER_SEQ}'] = []
  output_stats[f'{_PREFILL_INSERT}-{_MSEC_PER_SEQ}'] = []
  output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_TOK}'] = []
  output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_SEQ_PER_TOK}'] = []
  for (prefill_stats, prefill_insert_stats, ar_stats) in stats_list:
    if prefill_stats and _MSEC_PER_SEQ in prefill_stats:
      output_stats[f'{_PREFILL}-{_MSEC_PER_SEQ}'].append(prefill_stats[_MSEC_PER_SEQ])
    if prefill_insert_stats and _MSEC_PER_SEQ in prefill_insert_stats:
      output_stats[f'{_PREFILL_INSERT}-{_MSEC_PER_SEQ}'].append(prefill_insert_stats[_MSEC_PER_SEQ])
    if ar_stats and _MSEC_PER_TOK in ar_stats:
      output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_TOK}'].append(ar_stats[_MSEC_PER_TOK])
      output_stats[f'{_AUTOREGRESSIVE}-{_MSEC_PER_SEQ_PER_TOK}'].append(ar_stats[_MSEC_PER_SEQ_PER_TOK])
  # Result
  result = {}
  for k in output_stats.keys():
    if output_stats[k] and len(output_stats[k]) > 1:
      result[k] = get_quantiles(k, output_stats[k])
  return result

def tokens_to_text(tokenizer, sampled_tokens_list, batch_size):
  text_output = []
  for slot in range(batch_size):
    results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
    text = tokenizer.detokenize(results)
    # print(f"slot {slot} text:\n{text}")
    text_output.append(text)
  return text_output

def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  # Set up initial state
  decode_state = engine.init_decode_state()
  _, cache_size, _ = max_utils.summarize_pytree_data(decode_state["cache"], name="Cache")
  num_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")
  batch_size = engine.max_concurrent_decodes

  # Read the dataset
  dataset = read_openorca_dataset(_DATAFILE)
  dataset_size = len(dataset)
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  # Benchmark
  stats = []
  output_text = []
  print(f"\nBenchmarks started: dataset_size {dataset_size} and batch_size {batch_size}")
  for start_row in range(0, dataset_size, batch_size):
    decode_state = engine.init_decode_state()
    end_row = min(start_row + batch_size, dataset_size)
    prefill_tokens = {}
    prefill_true_lengths = {}
    count = 0
    for j in range(start_row, end_row):
      text = dataset[j]['prompt']
      batch_size = end_row - start_row
      prefill_tokens[count], prefill_true_lengths[count] = token_utils.tokenize_and_pad(
        text, vocab, is_bos=True, prefill_lengths=[_PREFILL_LENGTH])
      if count == 0:
        print(f'\nProcessing batch of size {batch_size} consisting of rows {start_row} to {end_row}')
        #print(f'Sample prompt: {text}')
      count+=1
    batch_stats, output_tokens = benchmark(config, engine, decode_state, params, prefill_tokens, prefill_true_lengths, start_row)
    batch_output_text = tokens_to_text(vocab.tokenizer, output_tokens, batch_size)
    rouge_val = eval_accuracy(batch_output_text, [dataset[j]['output'] for j in range(start_row, end_row)])
    output_text.extend(batch_output_text)
    stats.append(batch_stats)
    if _PRINT_BATCH_STATS:
      (prefill_stats, prefill_insert_stats, ar_stats) = batch_stats
      print(f'\nPrefill: {prefill_stats}')
      print(f'\nPrefill-Insert: {prefill_insert_stats}')
      print(f'\nAutoregressive: {ar_stats}')
      print(f'\nAccuracy: {rouge_val}')


  # Aggregate stats
  print('\nAggregating stats')
  result = aggregate_stats(stats)

  # Output stats
  print('\nResults')
  for k in result.keys():
    print('\t' + f'{result[k]}')

  # Output overall rouge score
  rouge_val = eval_accuracy(output_text, [dataset[j]['output'] for j in range(len(dataset))])
  print(f'\nAccuracy: {rouge_val}')


  print('\nEnd')
  # TBD - add rouge score

if __name__ == "__main__":
  pyconfig.initialize(sys.argv)
  main(pyconfig.config)