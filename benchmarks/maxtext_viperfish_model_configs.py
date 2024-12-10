"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
"""Shared Benchmark config for v6e orchestrations."""

import dataclasses
import typing
import xla_flags_library

from model_configs import MaxTextModel


llama2_70b_4096_int8 = MaxTextModel(
    model_name="llama2-70b-4096-int8",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 4,
        "remat_policy": "full",
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "remat_policy": "save_dot_except_mlpwi",
        "profiler": "xplane",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "tokenizer_path": "assets/tokenizer.llama2",
        "quantization": "int8",
        "steps": 100,
    },
    xla_flags=(
        xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

llama2_70b_4096_int8_ckp = MaxTextModel(
    model_name="llama2-70b-4096-int8",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 4,
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "reuse_example_batch": 1,
        "enable_checkpointing": True,
        "checkpoint_period": 200,
        "remat_policy": "save_dot_except_mlpwi",
        "profiler": "xplane",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "tokenizer_path": "assets/tokenizer.llama2",
        "quantization": "int8",
        "steps": 100,
    },
    xla_flags=(
        xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

llama2_70b_4096_real_data_int8 = MaxTextModel(
    model_name="llama2-70b-4096-rd-int8",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 4,
        "remat_policy": "full",
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "remat_policy": "save_dot_except_mlpwi",
        "profiler": "xplane",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "tfds",
        "tokenizer_path": "assets/tokenizer.llama2",
        "quantization": "int8",
        "steps": 100,
    },
    xla_flags=(
        xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

llama2_70b_4096_real_data = MaxTextModel(
    model_name="llama2-70b-4096-rd",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 4,
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "remat_policy": "save_dot_except_mlpwi",
        "profiler": "xplane",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "tfds",
        "tokenizer_path": "assets/tokenizer.llama2",
        "steps": 100,
    },
    xla_flags=(
        xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

_DENSE_VMEM_LIMIT=32768

DENSE_VMEM_LIMIT_FLAG = f" --xla_tpu_scoped_vmem_limit_kib={_DENSE_VMEM_LIMIT}"

llama3_1_405b_8192_fsdp_dcn = MaxTextModel(
    model_name="llama3-1-405b-8192-fsdp-dcn",
    model_type="llama3.1-405b",
    tuning_params={
        "per_device_batch_size": 1,
        "remat_policy": "full",
        "ici_fsdp_parallelism": -1,
        "ici_tensor_parallelism": 8,
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "steps": 100,
    },
    xla_flags=(
        xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + DENSE_VMEM_LIMIT_FLAG
    ),
)
# List of all models
maxstar_models = [
    llama2_70b_4096_int8,
    llama2_70b_4096_int8_ckp,
    llama2_70b_4096_real_data,
    llama3_1_405b_8192_fsdp_dcn,
]
