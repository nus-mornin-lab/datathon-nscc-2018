ARG CUDA=9.1
FROM nvidia/cuda:$CUDA-cudnn7-devel-ubuntu16.04 as base
WORKDIR /
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:marutter/rrutter3.5 && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        liblapack-dev \
        libopenblas-dev \
        libcurl4-openssl-dev \
        libjemalloc-dev \
        libopencv-dev \
        libssh2-1-dev \
        libssl-dev \
        libxml2-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        libgoogle-glog-dev \
        libiomp-dev \
        libsnappy-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgflags-dev \
        libsqlite3-dev \
        graphviz \
        pkg-config \
        cmake \
        vim \
        rsync \
        curl \
        wget \
        git \
        mercurial \
        zip \
        unzip \
        zlib1g-dev \
        python3.6-dev \
        r-base-dev \
        r-recommended && \
    curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm -f get-pip.py
RUN pip install --no-cache-dir cython cmake numpy pyyaml cffi future protobuf

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && \
    ldconfig
ENV BAZEL_VERSION 0.12.0
WORKDIR /bazel
RUN curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> /etc/bazel.bazelrc
WORKDIR /
ARG COMPUTE_CAPABILITIES=6.1,7.0
ENV CI_BUILD_PYTHON=python3.6 \
    TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=$CUDA_VERSION \
    TF_NEED_GCP=0 \
    TF_NEED_S3=0 \
    TF_NEED_KAFKA=0 \
    TF_CUDNN_VERSION=7 \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    TF_CUDA_COMPUTE_CAPABILITIES=$COMPUTE_CAPABILITIES
RUN git clone --branch=r1.8 --depth=1 https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package . && \
    pip install --no-cache-dir tensorflow*.whl && \
    cd / && rm -rf /tensorflow /usr/local/cuda/lib64/libcuda.so*

RUN git clone --branch=v0.72 --depth=1 --recursive https://github.com/dmlc/xgboost.git && \
    cd xgboost && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON && make -j && \
    cd ../python-package && \
    python3.6 setup.py install && \
    cd .. && rm -rf build && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make -j install && \
    cd / && rm -rf /xgboost

RUN git clone --depth=1 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    python3.6 setup_caffe2.py install && \
    cd / && rm -rf /pytorch

RUN git clone --branch=v0.7.6 --depth=1 https://github.com/Theano/libgpuarray.git && \
    cd libgpuarray && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j && make install && \
    ldconfig && \
    cd .. && \
    python3.6 setup.py build && python3.6 setup.py install && \
    cd / && rm -rf /libgpuarray
ENV PYCUDA_VERSION 2017.1.1
RUN pip download --no-cache-dir --no-deps pycuda && \
    tar xvf pycuda-$PYCUDA_VERSION.tar.gz && \
    cd pycuda-$PYCUDA_VERSION && \
    python3.6 configure.py && \
    python3.6 setup.py install && \
    cd / && rm -rf pycuda-$PYCUDA_VERSION*

COPY make.inc /
RUN hg clone https://bitbucket.org/icl/magma && cd magma && \
    mv /make.inc . && \
    make -j lib && make -j sparse-lib && make install prefix=/usr/local/magma && \
    echo "/usr/local/magma/lib" >> /etc/ld.so.conf.d/magma.conf && ldconfig

RUN git clone --branch=v0.4.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    NCCL_ROOT_DIR=/usr/lib/x86_64-linux-gnu python3.6 setup.py install && \
    cd / && rm -rf /pytorch

COPY mxnet_cuda_arch.patch /
RUN git clone --branch=v1.2.0 --depth=1 --recursive https://github.com/apache/incubator-mxnet mxnet && \
    cd mxnet && \
    patch < /mxnet_cuda_arch.patch && \
    make -j USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
        USE_NCCL=1 USE_NCCL_PATH=/usr/lib/x86_64-linux-gnu && \
    cd python && python3.6 setup.py install && \
    cd / && rm -rf /mxnet mxnet_cuda_arch.patch

COPY packages.r python-packages.txt /
RUN rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r python-packages.txt && \
    Rscript packages.r && \
    git clone --branch=0.8.11 --depth=1 --recursive https://github.com/IRkernel/IRkernel.git && \
    Rscript -e "devtools::install_local('IRkernel'); IRkernel::installspec(user = FALSE)" && \
    rm -rf IRkernel packages.r python-packages.txt

WORKDIR /opt/bin
ENV PATH /opt/bin:$PATH
RUN echo "!#/bin/bash" >> python && \
    echo 'python3.6 "$@"' >> python && \
    chmod +x python
WORKDIR /

EXPOSE 8888
