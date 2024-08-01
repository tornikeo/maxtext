export BRANCH="main"
export CMD=""
#export CMD="echo"
export CKPT_SUBDIR="mlperf_070924"
export KV_AXIS="dkv"

run_script () {
  export OPTIONS="MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2  checkpoint_is_quantized=True max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-70b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11  async_checkpointing=false attention=dot_product  inference_microbenchmark_prefill_lengths=1024"
  export RUN_NAME="${CMD}_microbenchmark_20240714_${BRANCH}_${QUANT}_${QUANT_CFG}_kv_${KV}_${KV_AXIS}"
  export LOAD_PARAMS_PATH="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/${CKPT_SUBDIR}/${QUANT}_${QUANT_CFG}"
  export QUANT_CFG_PATH="MaxText/configs/quantization/${QUANT_CFG}.json"

  MAXTEXT_COMMIT_ID=$(git log |head -n 1)
  JETSTREAM_COMMIT_ID=$(cd ~/JetStream; git log |head -n 1)
  DATASET=$(cat ~/JetStream/benchmarks/benchmark_serving.py  |grep pkl|grep orca_gpt4)
  BRANCH_STR=$(git branch)
  echo maxtext commit id: $MAXTEXT_COMMIT_ID
  echo jetstream commit id : $JETSTREAM_COMMIT_ID
  echo dataset: $DATASET
  echo branch: $BRANCH_STR
  echo
  echo "RUNNING  tee -a /tmp/${RUN_NAME}.out"
  ${CMD} python MaxText/inference_microbenchmark.py ${OPTIONS} quantization=${QUANT} quantize_kvcache=${KV} load_parameters_path=${LOAD_PARAMS_PATH} quant_cfg_path=${QUANT_CFG_PATH} kv_quant_axis=${KV_AXIS}| tee -a /tmp/${RUN_NAME}.out
}

export KV=False

export QUANT="int8"
export QUANT_CFG=""
run_script


export QUANT="int8w"
export QUANT_CFG=""
run_script


export QUANT="intmp"
export QUANT_CFG="mp_scale"
run_script

export KV=True

export QUANT="int8"
export QUANT_CFG=""
run_script


export QUANT="int8w"
export QUANT_CFG=""
run_script


export QUANT="intmp"
export QUANT_CFG="mp_scale"
run_script

