# Dockerfile for [NUS-MIT Datathon 2018](http://www.nus-datathon.com) server setup @ nscc

### Run `jupyter`

```sh
docker run \
    --runtime nvidia \
    -u $UID:$GID \
    -v /home:/home \
    -e USER \
    -e HOME \
    -w $HOME \
    -p 8888:8888 \
    --rm -it \
    datathon:cuda9.1-ubuntu16.04 \
    /bin/bash -c \
    "jupyter lab \
        --LabApp.open_browser=False \
        --LabApp.ip='0.0.0.0'"
```

Run `jupyter` @ nscc

```sh
nvidia-docker-run --net=host kiend/datathon-nscc:cuda9.0-ubuntu16.04 << EOF
    export HOME=$HOME
    export USER=$USER
    cd $HOME
    jupyter lab --LabApp.ip='0.0.0.0'
EOF
```

### Python
Default `python` (`/opt/bin/python`) uses `python 3.6.5`.

#### Pre-installed frameworks:
- `tensorflow`
- `pytorch`
- `mxnet`
- `caffe`*
- `caffe2`*
- `theano`
- `keras` (`keras-mxnet`)
- `pymc3`
- Packages in the pydata stack: `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`, ... Run `pip list` for the full list.

To install other packages: `pip install --user <package_name>`

\* *To use `caffe` or `caffe2`, first run `import caffe_path` or `import caffe2_path`. i.e.*

```python
import caffe_path
import caffe
```
*or*
```python
import caffe2_path
import caffe2
```

### R
#### Pre-installed packages:
- `tidyverse`
    - `dplyr`
    - `tidyr`
    - `ggplot2`
    - ...
- `caret`
- `rjags`
- `tensorflow`
- `keras`
- ...

To install other packages: `install.packages("<packages_name>")`
