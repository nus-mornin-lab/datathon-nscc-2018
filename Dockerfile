ARG CUDA=9.1
FROM nvidia/cuda:$CUDA-cudnn7-devel-ubuntu16.04 as base
WORKDIR /
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:marutter/rrutter3.5 && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        python3.6-dev \
        git \
        curl \
        wget \
        rsync \
        vim \
        r-base-dev \
        r-recommended && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm -f get-pip.py

FROM base as mxnet
WORKDIR /
COPY mxnet_cuda_arch.patch /
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjemalloc-dev \
    libopencv-dev
RUN git clone --branch=v1.2.0 --depth=1 --recursive https://github.com/apache/incubator-mxnet mxnet && \
    cd mxnet && \
    patch < /mxnet_cuda_arch.patch && \
    pip install --no-cache-dir cython && \
    make -j USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
        USE_NCCL=1 USE_NCCL_PATH=/usr/lib/x86_64-linux-gnu

FROM base as tf-runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && \
    ldconfig

FROM tf-runtime as tf-devel
ENV BAZEL_VERSION 0.12.0
WORKDIR /bazel
RUN curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> /etc/bazel.bazelrc && \
    pip install --no-cache-dir numpy

FROM tf-devel as tf
ARG COMPUTE_CAPABILITIES=6.1,7.0
WORKDIR /
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
    mkdir /dist && bazel-bin/tensorflow/tools/pip_package/build_pip_package /dist && \
    cd / && rm -rf /tensorflow

FROM base as xgboost
WORKDIR /
RUN git clone --branch=v0.72 --depth=1 --recursive https://github.com/dmlc/xgboost.git && \
    cd xgboost && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON && make -j && \
    cd ../python-package && \
    mkdir /dist && python3.6 setup.py bdist_wheel -d /dist && \
    cd .. && rm -rf build && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make -j install && \
    mv R-package /r && \
    cd / && rm -rf /xgboost

FROM base as caffe2
WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev \
        libiomp-dev \
        libsnappy-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgflags-dev \
        graphviz && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir numpy future protobuf && \
    git clone --depth=1 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    mkdir /dist && python3.6 setup_caffe2.py bdist_wheel -d /dist && \
    cd / && rm -rf /pytorch

FROM base as theano
WORKDIR /libgpuarray
RUN mkdir /dist && \
    pip install --no-cache-dir cython numpy && \
    git clone --branch=v0.7.6 --depth=1 https://github.com/Theano/libgpuarray.git . && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j && make install && \
    ldconfig && \
    cd .. && \
    python3.6 setup.py bdist_wheel -d /dist
ENV PYCUDA_VERSION 2017.1.1
WORKDIR /
RUN pip download --no-cache-dir --no-deps pycuda && \
    tar xvf pycuda-$PYCUDA_VERSION.tar.gz && \
    cd pycuda-$PYCUDA_VERSION && \
    python3.6 configure.py && \
    python3.6 setup.py bdist_wheel -d /dist && \
    cd / && rm -rf pycuda-$PYCUDA_VERSION*

FROM base as magma
WORKDIR /magma
RUN apt-get update && apt-get install -y --no-install-recommends mercurial && \
    rm -rf /var/lib/apt/lists/* && \
    hg clone https://bitbucket.org/icl/magma . && \
    wget https://gist.githubusercontent.com/kiendang/3931760719cf37cbf22355900e89d57a/raw/ab632aebaaba1c1b8a733dc2014a51f1dbf69faf/make.inc && \
    make -j lib && make -j sparse-lib

FROM base as pytorch
COPY --from=magma /magma /magma
WORKDIR /
RUN cd /magma && make install prefix=/usr/local/magma && cd / && rm -rf /magma && \
    echo "/usr/local/magma/lib" >> /etc/ld.so.conf.d/magma.conf && ldconfig && \
    pip install --no-cache-dir cmake cffi pyyaml && \
    git clone --branch=v0.4.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    mkdir /dist && NCCL_ROOT_DIR=/usr/lib/x86_64-linux-gnu python3.6 setup.py bdist_wheel -d /dist && \
    cd / && rm -rf /pytorch

FROM base
COPY --from=tf /dist/*.whl /packages/python/
COPY --from=xgboost /dist/*.whl /packages/python/
COPY --from=xgboost /r /packages/r/xgboost
COPY --from=mxnet /dist/*.whl /packages/python/
COPY --from=caffe2 /dist/*.whl /packages/python/
COPY --from=theano /libgpuarray /libgpuarray
COPY --from=theano /dist/*.whl /packages/python/
COPY --from=magma /magma /magma
COPY --from=pytorch /dist/*.whl /packages/python/
WORKDIR /
COPY packages.r python-packages.txt /
RUN cd /magma && make install prefix=/usr/local/magma && cd / && rm -rf /magma && \
    cd /libgpuarray/build && make install && cd / && rm -rf /libgpuarray && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
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
        tmux \
        emacs && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/usr/local/magma/lib" >> /etc/ld.so.conf.d/magma.conf && ldconfig && \
    find packages -name '*.whl' | xargs pip install --no-cache-dir && \
    pip install --no-cache-dir -r python-packages.txt && \
    Rscript packages.r && \
    git clone --branch=0.8.11 --depth=1 --recursive https://github.com/IRkernel/IRkernel.git && \
    Rscript -e "devtools::install_local('IRkernel'); IRkernel::installspec(user = FALSE)" && \
    rm -rf IRkernel packages.r python-packages.txt
EXPOSE 8888
