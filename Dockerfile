FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timez

RUN adduser --disabled-password --gecos "" ana && \
    echo 'ana ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers \

USER ana

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN pip install jupyter notebook
RUN pip install poetry-kernel

WORKDIR /home/ana/llm-sre
COPY --chown=nonroot:nonroot . /home/ana/llm-sre
RUN chmod -R 755 /home/ana/llm-sre

RUN poetry env use $(which python3.10)
RUN poetry config virtualenvs.create false
RUN poetry install 

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=1" FORCE_CMAKE=1 LLAMA_CUBLAS=1 poetry run pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

EXPOSE 8888
CMD jupyter notebook --no-browser --port 8888 --ip=* --allow-root

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/