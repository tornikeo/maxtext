CMD="echo"
#CMD=""

CKPT_SUBDIR="post_mlperf"
BRANCH="msingh-aqt"
run_benchmark () {
  cd ~/maxtext
  OPTIONS="MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2  checkpoint_is_quantized=True max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-70b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 async_checkpointing=false attention=dot_product  inference_microbenchmark_prefill_lengths=1024"
  RUN_NAME="${BRANCH}_${QUANT}_${QUANT_CFG}_kv_${KV}_bs_${BATCH_SIZE}"
  LOAD_PARAMS_PATH="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/${CKPT_SUBDIR}/${QUANT}_${QUANT_CFG}"

  MAXTEXT_COMMIT_ID=$(git log |head -n 1)
  BRANCH_STR=$(git branch)
  OUTDIR=~/logs/${CMD}_microbenchmarks/${KV}/${BATCH_SIZE}
  mkdir -p ${OUTDIR}
  OUTFILE=${OUTDIR}/${RUN_NAME}.out
  rm -f $OUTFILE
  echo
  echo maxtext commit id: $MAXTEXT_COMMIT_ID |tee -a $OUTFILE
  echo
  echo branch: $BRANCH_STR |tee -a $OUTFILE
  echo
  echo run_name: ${RUN_NAME}_MS |tee -a $OUTFILE
  echo
  echo "RUNNING  with output directed to ${OUTFILE}"
  ${CMD} python MaxText/inference_microbenchmark.py ${OPTIONS} per_device_batch_size=${BATCH_SIZE} quantization=${QUANT} quantize_kvcache=${KV} load_parameters_path=${LOAD_PARAMS_PATH} quant_cfg_path=${QUANT_CFG_PATH}| grep -v arning |tee -a $OUTFILE
}

run_benchmarks_int8 () {
  QUANT_CFG=""
  QUANT_CFG_PATH=""
  VAL_LIST="int8 int8w"
  for value in $VAL_LIST
  do
    QUANT=$value
    run_benchmark
  done
}

run_benchmarks_int4 () {
  QUANT_CFG=""
  QUANT_CFG_PATH=""
  VAL_LIST="int4 int4_8 int4w"
  for value in $VAL_LIST
  do
    QUANT=$value
    run_benchmark
  done
}

run_benchmarks_intmp () {
  QUANT_CFG="mp_scale"
  QUANT_CFG_PATH="MaxText/configs/quantization/${QUANT_CFG}.json"
  VAL_LIST="intmp8 intmp intmp4"
  for value in $VAL_LIST
  do
    QUANT=$value
    run_benchmark
  done
}

KV=True
BATCH_SIZES="20 40 48 54 60"
for bs in $BATCH_SIZES
do
  BATCH_SIZE=$bs
  run_benchmarks_intmp
done

