#!/usr/bin/env bash

# Example:
# bash offline_run_llama.sh -r drq -c mp_v0 -u user_5000.conf -n

dry_run=false
run_name='$(date +'%Y%m%d%H%M%S')'
mp_config="mp_scale"
user_conf="user.conf"

while getopts "nr:c:u:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      r ) run_name="$OPTARG" ;;
      c ) mp_config="$OPTARG" ;;
      u ) user_conf="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi


export BASEDIR=/home/msingh/inference_mlperf4.1
export USER_CONFIG=/home/msingh/maxtext/MaxText/${user_conf}
export DATA_DISK_DIR=/home/msingh/loadgen_run_data
export API_URL=0.0.0.0:9000
export DATASET_TYPE=full
export DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
export PERF_SAMPLE_COUNT=1000
export TOTAL_SAMPLE_COUNT=24576
export LOG_INTERVAL=900
export MODEL_NAME=llama70b
export BATCH_SIZE_EXP=8
# HF model id
export BATCH_AND_PREFILL_LEN="256,96|512,48|1024,24"
export QUANT="intmp"
export QUANT_CFG="configs/quantization/${mp_config}.json"
export SAVE_QUANT_PARAMS_PATH="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/${run_name}/${QUANT}_${mp_config}"
export MAXENGINE_ARGS=" quantization=${QUANT} quant_cfg_path=${QUANT_CFG} load_parameters_path=${SAVE_QUANT_PARAMS_PATH} "

LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS


run_loadgen() {

  OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${QUANT}-${mp_config}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}
  OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
  mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}
  OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json


  echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
  echo "DATASET_PATH: ${DATASET_PATH}"
  echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
  echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
  echo "USER_CONFIG: ${USER_CONFIG}"
  echo "BATCH_AND_PREFILL_LEN: ${BATCH_AND_PREFILL_LEN}"
  echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"

  ${cmd} python -m offline_mode \
    --mlperf_test_mode=${TEST_MODE} \
	  --input_mode tokenized \
    --output_mode tokenized \
	  --mlperf_conf $BASEDIR/mlperf.conf \
	  --user_conf ${USER_CONFIG} \
	  --audit_conf ${AUDIT_CONF} \
	  --total_sample_count ${TOTAL_SAMPLE_COUNT} \
	  --dataset_path ${DATASET_PATH} \
    --prefill_lengths_and_batch_sizes ${BATCH_AND_PREFILL_LEN} \
    --maxengine_args ${MAXENGINE_ARGS} \
	  --output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/${LOADGEN_RUN_TYPE}_log.log

}

run_loadgen_performance () {
  LOADGEN_RUN_TYPE=offline-performance
  TEST_MODE="performance"
  AUDIT_CONF="no_audit"
  run_loadgen
}

run_loadgen_audit () {
  LOADGEN_RUN_TYPE=offline-audit
  TEST_MODE="performance"
  AUDIT_CONF="$BASEDIR/compliance/nvidia/TEST06/audit.config"
  run_loadgen
}

run_loadgen_accuracy () {
  LOADGEN_RUN_TYPE=offline-accuracy
  TEST_MODE="accuracy"
  AUDIT_CONF="no_audit"
  run_loadgen

  # Eval Run
  if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
    ${CMD} python3 evaluate-accuracy.py \
      --checkpoint-path meta-llama/Llama-2-70b-chat-hf \
      --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
      --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
  fi
}


# echo
# echo "Starting loadgen performance run"
# run_loadgen_performance

# echo
# echo "Starting loadgen audit"
# run_loadgen_audit

echo
echo "Starting loadgen accuracy"
run_loadgen_accuracy



