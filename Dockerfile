FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install albumentations

RUN mkdir /data
VOLUME /data

RUN mkdir /opt/app
WORKDIR /opt/app

COPY *.py /opt/app/

CMD ["/bin/bash"]