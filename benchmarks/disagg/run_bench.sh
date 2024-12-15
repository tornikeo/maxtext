pushd /

gsutil cp gs://jwyang-data/meta_original_llama2_openorca_results/llama13b_chat_openorca_input.json .

export dataset_path=llama13b_chat_openorca_input.json

# Run a few requests as warmup
echo "============================"
echo "Run a few requests as warmup"
echo "============================"
python3 JetStream/benchmarks/benchmark_serving.py \
  --tokenizer maxtext/assets/tokenizer.llama2 \
  --warmup-mode sampled \
  --save-result \
  --save-request-outputs \
  --request-outputs-file-path outputs.json \
  --request-rate 1 \
  --num-prompts 3 \
  --max-output-length 1024 \
  --dataset openorca \
  --dataset-path ${dataset_path} \
  --server=0.0.0.0 

# Actually run benchmarking

echo "================"
echo "Run benchmarking
echo "================
python3 JetStream/benchmarks/benchmark_serving.py \
  --tokenizer maxtext/assets/tokenizer.llama2 \
  --save-result \
  --save-request-outputs \
  --request-outputs-file-path outputs.json \
  --request-rate 20 \
  --num-prompts 1000 \
  --max-output-length 1024 \
  --dataset openorca \
  --dataset-path ${dataset_path} \
  --server=0.0.0.0 

