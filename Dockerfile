# Dockerfile for testing pegasusio

FROM debian:buster-slim
SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends \
        build-essential \
        gnupg \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip

RUN pip3 install setuptools --no-cache-dir && \
    pip3 install cython --no-cache-dir

RUN apt-get -qq -y remove curl gnupg && \
    apt-get -qq -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY . /pegasusio/
WORKDIR /pegasusio/tests
RUN git clone https://github.com/klarman-cell-observatory/pegasusio-test-data.git
WORKDIR /pegasusio/
RUN pip3 install -e .

WORKDIR /pegasusio/tests
