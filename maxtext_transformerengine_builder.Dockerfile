FROM ghcr.io/nvidia/jax:maxtext-2024-12-11

WORKDIR /root
ENV NVTE_FRAMEWORK=jax


RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git pull
RUN git checkout e5edd6c
RUN git submodule update --init --recursive
RUN python setup.py bdist_wheel
