FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

WORKDIR /workspace

RUN pip install --no-cache-dir transformers accelerate numpy

COPY . /workspace/
