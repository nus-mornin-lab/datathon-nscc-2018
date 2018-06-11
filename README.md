# `Dockerfile` for [NUS-MIT Datathon 2018](http://www.nus-datathon.com) server setup @ nscc

Run `jupyter`

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
