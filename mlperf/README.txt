commands before you run llama_offline_performance.sh


git clone -b msingh-mp https://github.com/google/maxtext.git

source .env/bin/activate
cd maxtext && bash setup.sh MODE=stable && cd ..


1. install loadgen

git clone -b mlperf4.1-jetstream https://github.com/tpu-inference/inference_mlperf4.1.git

or



cd loadgen/ && pip install .

2. pip install

pip install \
transformers==4.31.0 \
nltk==3.8.1 \
evaluate==0.4.0 \
absl-py==1.4.0 \
rouge-score==0.1.2 \
sentencepiece==0.1.99 \
accelerate==0.21.0

3. get datasets

mkdir -p ~/loadgen_run_data
cd ~/loadgen_run_data

gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl .
mv open_orca_gpt4_tokenized_llama.calibration_1000.pkl processed-calibration-data.pkl

gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl .
mv open_orca_gpt4_tokenized_llama.sampled_24576.pkl processed-data.pkl


huggingface-cli login --token ${HF_TOKEN}

cd inference_mlperf4.1/language/llama2-70b/tpu/


# Create a user.conf file per your requirements in the inference_mlperf4.1/language/llama2-70b/tpu/ directory

*.Server.target_qps = 1.5
*.Offline.target_qps = 20


cd ~/maxtext/Maxtext
bash offline_run_llama.sh

