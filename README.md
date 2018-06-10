# `Dockerfile` for [NUS-MIT Datathon 2018](http://www.nus-datathon.com) `nscc` server setup

Running `jupyter`

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
