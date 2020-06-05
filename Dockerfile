# Dockerfile for testing pegasusio

FROM debian:buster-slim
SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends \
        build-essential \
        gnupg \
        curl \
        python3 \
        python3-dev \
        python3-pip

RUN pip3 install setuptools==47.1.1 --no-cache-dir && \
    pip3 install cython==0.29.19 --no-cache-dir && \
    pip3 install numpy==1.18.5 --no-cache-dir && \
    pip3 install scipy==1.4.1 --no-cache-dir && \
    pip3 install pandas==1.0.4 --no-cache-dir && \
    pip3 install anndata==0.7.3 --no-cache-dir && \
    pip3 install loompy==3.0.6 --no-cache-dir && \
    pip3 install docopt==0.6.2 --no-cache-dir && \
    pip3 install natsort==7.0.1 --no-cache-dir && \
    pip3 install importlib-metadata==1.6.0 --no-cache-dir && \
    pip3 install zarr==2.4.0 --no-cache-dir

RUN apt-get -qq -y remove curl gnupg && \
    apt-get -qq -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

COPY . /pegasusio/
WORKDIR /pegasusio/tests
RUN git clone https://github.com/klarman-cell-observatory/pegasus-test-data.git
WORKDIR /pegasusio/
RUN pip install -e .
