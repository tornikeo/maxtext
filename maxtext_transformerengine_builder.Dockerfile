FROM ghcr.io/nvidia/jax:base

WORKDIR /root
ENV NVTE_FRAMEWORK=jax


RUN git clone https://github.com/NVIDIA/TransformerEngine
WORKDIR /root/TransformerEngine
RUN git checkout e5edd6cc3d5a868bb3fe4e81088d22aab505a30d
RUN git submodule update --init --recursive
RUN python setup.py bdist_wheel
