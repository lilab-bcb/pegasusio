# Dockerfile for testing pegasusio

FROM debian:buster-slim
SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends \
        build-essential \
        git \
        python3 \
        python3-dev \
        python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip && \
    python -m pip install setuptools wheel

RUN apt-get -qq -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

COPY . /pegasusio/
WORKDIR /pegasusio/tests
RUN git clone https://github.com/klarman-cell-observatory/pegasusio-test-data.git
WORKDIR /pegasusio/
RUN python -m pip install -e .

WORKDIR /pegasusio/tests
