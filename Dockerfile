FROM nvidia/cuda:12.1.1-devel-ubuntu20.04 

ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timez

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN pip install jupyter notebook
RUN pip install poetry-kernel

WORKDIR /llama

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install

EXPOSE 8888
CMD jupyter notebook --no-browser --port 8888 --ip=* --allow-root