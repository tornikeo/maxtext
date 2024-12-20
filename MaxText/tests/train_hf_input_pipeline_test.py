"""
Copyright 2023 Google LLC

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

"""Short test for train.py with TFDS c4, using HF"""
import os
import unittest
import pytest
from train import main as train_main
from absl.testing import absltest


class Train(unittest.TestCase):
  """Tests base config using HF input pipeline"""

  # Shared parameters
  CONFIG = [
      None,
      "configs/base.yml",
      r"base_output_directory=gs://runner-maxtext-logs",
      "run_name=runner_test",
      "steps=2",
      "enable_checkpointing=False",
      "dataset_type=hf",
      "hf_path=parquet",
      r"hf_train_files=gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet",
      r"tokenizer_path=google-t5/t5-large",
  ]

  @pytest.mark.tpu_only
  def test_default_config(self):
    train_main(Train.CONFIG)

  def test_default_config_dot_product(self):
    train_main(Train.CONFIG + ["attention=dot_product"])


if __name__ == "__main__":
  absltest.main()
