ARG CUDA=9.1
FROM nvidia/cuda:$CUDA-cudnn7-devel-ubuntu16.04 as base
WORKDIR /
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:marutter/rrutter3.5 && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        liblapack-dev \
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
        libgtest-dev \
        libiomp-dev \
        libopenmpi-dev \
        libsnappy-dev \
        liblmdb-dev \
        libleveldb-dev \
        libboost-all-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgflags-dev \
        libsqlite3-dev \
        openmpi-bin \
        openmpi-doc \
        graphviz \
        jags \
        pkg-config \
        cmake \
        less \
        vim \
        emacs \
        rsync \
        curl \
        wget \
        tmux \
        ed \
        git \
        mercurial \
        zip \
        unzip \
        zlib1g-dev \
        python3.6-dev \
        r-base-dev \
        r-recommended && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bin
ENV PATH /opt/bin:$PATH
RUN echo "#!/bin/bash" >> python && \
    echo 'python3.6 "$@"' >> python && \
    chmod +x python && \
    curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm -f get-pip.py && \
    pip install --no-cache-dir cython cmake numpy pyyaml cffi future protobuf

ENV BAZEL_VERSION 0.14.1
WORKDIR /opt/bazel
RUN curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o ./LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >> /etc/bazel.bazelrc

ARG COMPUTE_CAPABILITIES=6.1,7.0
ENV TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=$CUDA_VERSION \
    TF_CUDNN_VERSION=7 \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    TF_CUDA_COMPUTE_CAPABILITIES=$COMPUTE_CAPABILITIES
WORKDIR /tensorflow
RUN git clone --branch=r1.8 --depth=1 https://github.com/tensorflow/tensorflow.git .
COPY tensorflow_nasm_urls.patch ./
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    patch -p1 < tensorflow_nasm_urls.patch && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package . && \
    pip install --no-cache-dir tensorflow*.whl
WORKDIR /
RUN rm -rf tensorflow /usr/local/cuda/lib64/libcuda.so.1

WORKDIR /xgboost
RUN git clone --branch=release_0.72 --depth=1 https://github.com/dmlc/xgboost.git . && \
    git submodule update --init -j $(( $(nproc) + 1 )) && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON && make -j && \
    cd ../python-package && \
    python setup.py install && \
    cd .. && rm -rf build && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make -j install
WORKDIR /
RUN rm -rf xgboost

WORKDIR /caffe2
RUN git clone --depth=1 https://github.com/pytorch/pytorch.git . && \
    git submodule update --init -j $(( $(nproc) + 1 )) && \
    mkdir build && cd build && \
    cmake .. && make -j install
WORKDIR /
RUN rm -rf caffe2 && \
    echo "import sys\nsys.path.append('/usr/local/lib/python3/dist-packages')" >> \
        /usr/local/lib/python3.6/dist-packages/caffe2_path.py

WORKDIR /libgpuarray
RUN git clone --branch=v0.7.6 --depth=1 https://github.com/Theano/libgpuarray.git . && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j && make install && \
    ldconfig && \
    cd .. && \
    python setup.py build && python setup.py install
WORKDIR /
RUN rm -rf libgpuarray
ENV PYCUDA_VERSION 2017.1.1
RUN pip download --no-cache-dir --no-deps pycuda && \
    tar xvf pycuda-$PYCUDA_VERSION.tar.gz && \
    cd pycuda-$PYCUDA_VERSION && \
    python configure.py && \
    python setup.py install && \
    cd .. && rm -rf pycuda-*

WORKDIR /magma
RUN hg clone https://bitbucket.org/icl/magma .
COPY ["make.inc", "magma_codegen.patch", "./"]
RUN patch -p1 < magma_codegen.patch && \
    make -j lib && make -j sparse-lib && make install prefix=/usr/local/magma
WORKDIR /
RUN rm -rf magma && \
    echo "/usr/local/magma/lib" >> /etc/ld.so.conf.d/magma.conf && ldconfig

WORKDIR /pytorch
RUN git clone --depth=1 https://github.com/pytorch/pytorch.git . && \
    git submodule update --init -j $(( $(nproc) + 1 )) && \
    NCCL_ROOT_DIR=/usr/lib/x86_64-linux-gnu python setup.py install
WORKDIR /
RUN rm -rf pytorch

WORKDIR /mxnet
RUN git clone --branch=v1.2.0 --depth=1 https://github.com/apache/incubator-mxnet .
COPY mxnet_cuda_arch.patch ./
RUN git submodule update --init --recursive -j $(( $(nproc) + 1 )) && \
    patch -p1 < mxnet_cuda_arch.patch && \
    make -j USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 \
        USE_NCCL=1 USE_NCCL_PATH=/usr/lib/x86_64-linux-gnu && \
    cd python && python setup.py install
WORKDIR /
RUN rm -rf mxnet

WORKDIR /opt/caffe
RUN git clone --branch=1.0 --depth=1 https://github.com/BVLC/caffe.git .
COPY Makefile.config ./
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
       /usr/lib/x86_64-linux-gnu/libboost_python3.so && \
    ldconfig && \
    pip install --no-cache-dir scikit-image && \
    make -j all && \
    make distribute && \
    echo "import sys\nsys.path.append('/opt/caffe/python')" >> \
        /usr/local/lib/python3.6/dist-packages/caffe_path.py && \
    cp -r ./distribute/lib/* /usr/lib/ && \
    cp -r ./distribute/include/* /usr/include/ && \
    cp -r ./distribute/bin/* /usr/bin/ && \
    ldconfig

WORKDIR /
COPY packages.r python-packages.txt ./
RUN pip install --no-cache-dir -r python-packages.txt && \
    Rscript packages.r && \
    git clone --branch=0.8.11 --depth=1 --recursive https://github.com/IRkernel/IRkernel.git && \
    Rscript -e "devtools::install_local('IRkernel'); IRkernel::installspec(user = FALSE)" && \
    rm -rf IRkernel packages.r python-packages.txt

WORKDIR /opt/tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini ./tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini.asc ./tini.asc
RUN gpg --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 595E85A6B1B4779EA4DAAEC70B588DFF0527A9B7 && \
    gpg --verify ./tini.asc && \
    chmod +x ./tini

WORKDIR /
EXPOSE 8888
ENTRYPOINT ["/opt/tini/tini", "--"]
