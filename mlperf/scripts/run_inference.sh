export BRANCH=quant
export CMD="echo"
export CKPT_SUBDIR="mlperf_070924"

export QUANT="int8w"
export QUANT_CFG=""
export KV=True

run_inference () {
  export SCRIPT="${CMD} python MaxText/maxengine_server.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2  checkpoint_is_quantized=True max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-70b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11  async_checkpointing=false attention=dot_product "
  export RUN_NAME="${CMD}_inference_${BRANCH}_${QUANT}_${QUANT_CFG}_kv_${KV}"
  export LOAD_PARAMS_PATH="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/${CKPT_SUBDIR}/${QUANT}_${QUANT_CFG}"
  export QUANT_CFG_PATH="MaxText/configs/quantization/${QUANT_CFG}.json"

  echo
  echo "RUNNING | tee -a /tmp/${RUN_NAME}.out"
  ${SCRIPT} quantization=${QUANT} quantize_kvcache=${KV} load_parameters_path=${LOAD_PARAMS_PATH} |tee -a /tmp/${RUN_NAME}.out
}

run_benchmark () {
OUTFILE=/tmp/${CMD}_benchmark_${BRANCH}_${QUANT}_${QUANT_CFG}_kv_${KV}.txt
rm $OUTFILE
echo "RUNNING | tee -a $OUTFILE"
echo "Processes" >> $OUTFILE
echo $(ps -ef |grep python |grep -i maxtext) >> $OUTFILE
echo "Jetstream">> $OUTFILE
echo $(cd ~/JetStream; git log |head -n 1) >> $OUTFILE
echo "Jetstream">> $OUTFILE
echo $(cd ~/JetStream; git diff) >> $OUTFILE
echo "Maxtext">> $OUTFILE
echo $(cd ~/maxtext; git log |head -n 1) >> $OUTFILE
echo >> $OUTFILE
OPTIONS="--tokenizer maxtext/assets/tokenizer.llama2 --num-prompts 5000 --dataset openorca --request-rate 5 --max-output-length 1024  --run-eval True --warmup-mode sampled "
${CMD} python ~/JetStream/benchmarks/benchmark_serving.py  ${OPTIONS} |tee -a ${OUTFILE}
}


echo
run_inference


echo
run_benchmark
