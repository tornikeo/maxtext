echo "Running 1b.sh"
# 16B parameter model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5e/16b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v5e/16b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5e-256 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
set -e -x

export PLATFORM="gce"
export EXECUTABLE="train.py" # or train_compile.py

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# using gcsfuse
echo "Mounting bucket to /tmp/gcsfuse/"
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=aireenmei-us-central1 MOUNT_PATH=/tmp/gcsfuse
DATASET_PATH=/tmp/gcsfuse

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

JAX_PLATFORMS=tpu python3 MaxText/standalone_dataloader.py MaxText/configs/base.yml \
    steps=100 per_device_batch_size=8 \
    enable_checkpointing=False \
    max_target_length=2048 global_parameter_scale=1 \
    base_output_directory=gs://aireenmei-multipod/grain_scale \
    grain_train_files=/tmp/gcsfuse/maxtext-dataset/array-record/dummy/dataset-c-shard-10k/3.0.1/*.array_record \
    dataset_type=grain \
    grain_worker_count=4 \
    run_name=aireen-$(date +%Y%m%d-%H-%M)
    # profiler=xplane
    # skip_first_n_steps_for_profiler=10 \
    #grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-* \
    #grain_train_files=/tmp/gcsfuse/maxtext-dataset/array-record/dummy/dataset-c/c4-train*.array_record \
    # grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record-* \
    # grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation* \
    # eval_per_device_batch_size=4 eval_interval=10 eval_batch_num=5 \

#gsutil cp $HOME/gcsfuse.json ${OUTPUT_PATH}/${M_RUN_NAME}/
