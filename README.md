# `Dockerfile` for [NUS-MIT Datathon 2018](http://www.nus-datathon.com) `nscc` server setup

Running `jupyter`

```sh
docker run \
    --runtime nvidia \
    -u $UID:$GID \
    -v /home:/home \
    -p 0.0.0.0:8888:8888 \
    --rm -it \
    datathon:cuda9.1-ubuntu16.04 \
    /bin/bash -c \
    "export HOME=$HOME; \
    export USER=$USER; \
    jupyter lab \
        --LabApp.open_browser=False \
        --LabApp.ip='0.0.0.0' \
        --LabApp.notebook_dir=$HOME \
        --LabApp.token=''"
```