#!/usr/bin/env bash
me=$(basename "$0")


export BASEDIR=/home/msingh/inference_mlperf4.1
export USER_CONFIG=$BASEDIR/language/llama2-70b/tpu/user.conf
export DATA_DISK_DIR=/home/msingh/loadgen_run_data
export API_URL=0.0.0.0:9000
export DATASET_TYPE=full
export DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
export TOTAL_SAMPLE_COUNT=24576
export LOG_INTERVAL=900
export MODEL_NAME=llama70b
export BATCH_SIZE_EXP=8
# HF model id
export TOKENIZER_PATH="/home/msingh/maxtext/assets/tokenizer.llama2" #"meta-llama/Llama-2-70b-chat-hf"
export BATCH_AND_PREFILL_LEN="256,80|512,40|1024,20"
export QUANT="int4_8"
export QUANT_CFG="" #configs/quantization/mp_scale.json"
export SAVE_QUANT_PARAMS_PATH="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/post_mlperf/int4_8_"
export KVCACHE="True"
export MAXENGINE_ARGS="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH}  \
quantization=${QUANT} quant_cfg_path=${QUANT_CFG} quantize_kvcache=${KVCACHE} \
load_parameters_path=${SAVE_QUANT_PARAMS_PATH} \
compute_axis_order=0,1,2,3 ar_cache_axis_order=0,1,2,3 prefill_cache_axis_order=0,1,2,3"

LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}

mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

echo "LOADGEN_RUN_TYPE: ${LOADGEN_RUN_TYPE}"
echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "BATCH_SIZE_EXP: ${BATCH_SIZE_EXP}"
echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
echo "USER_CONFIG: ${USER_CONFIG}"
echo "BATCH_AND_PREFILL_LEN: ${BATCH_AND_PREFILL_LEN}"
echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"

CMD=""
CMD="echo"

echo
LOADGEN_RUN_TYPE=offline-performance
${CMD} python -m offline_mode \
        --mlperf_test_mode=performance \
	--input_mode tokenized \
        --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf no_audit \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
        --prefill_lengths_and_batch_sizes ${BATCH_AND_PREFILL_LEN} \
        --maxengine_args "${MAXENGINE_ARGS}" \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_performance_log.log

echo
LOADGEN_RUN_TYPE=offline-audit
${CMD} python -m offline_mode \
        --mlperf_test_mode=performance \
	--input_mode tokenized \
        --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf $BASEDIR/compliance/nvidia/TEST06/audit.config \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
        --prefill_lengths_and_batch_sizes ${BATCH_AND_PREFILL_LEN} \
        --maxengine_args "${MAXENGINE_ARGS}" \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_audit_log.log

echo
LOADGEN_RUN_TYPE=offline-accuracy
${CMD} python -m offline_mode \
        --mlperf_test_mode=accuracy \
	--input_mode tokenized \
        --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf no_audit \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
        --prefill_lengths_and_batch_sizes ${BATCH_AND_PREFILL_LEN} \
        --maxengine_args "${MAXENGINE_ARGS}" \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_accuracy_log.log
echo
# Eval Run
if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
        ${CMD} python3 evaluate-accuracy.py \
                --checkpoint-path meta-llama/Llama-2-70b-chat-hf \
                --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
                --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
fi
echo
