FROM simaai/ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install \
    build-essential ccache \
    wget curl awscli cmake unzip && \
    rm -rf /var/lib/apt/lists/*
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get -y update && \
  apt-get install -y python3-pip python3-dev && \
  rm -rf /var/lib/apt/lists/* && \
  python3 -m pip install --upgrade pip

ADD install_sonar.sh /tmp/
RUN /tmp/install_sonar.sh
ENV PATH=${PATH}:/opt/sonar-scanner/bin

RUN \
  curl https://letsencrypt.org/certs/lets-encrypt-x3-cross-signed.pem.txt  \
    -o /usr/local/share/ca-certificates/letsencryptx3.crt && \
  update-ca-certificates

ARG MLA_BRANCH="master"
ARG MLA_VERSION="0.0.1"
ADD mla-${MLA_VERSION}-Linux.deb /tmp
RUN apt-get update && apt-get install -y /tmp/mla-${MLA_VERSION}-Linux.deb && rm -rf /var/lib/apt/lists/*

ARG N2A_COMPILER_BRANCH="master"
ARG N2A_COMPILER_VERSION="0.0.1"
ADD sima_mlc-${N2A_COMPILER_VERSION}_${N2A_COMPILER_BRANCH}-py3-none-any.whl /tmp
RUN pip3 install /tmp/sima_mlc-${N2A_COMPILER_VERSION}_${N2A_COMPILER_BRANCH}-py3-none-any.whl

RUN apt-get update && \
  apt-get install -y \
    libopenblas-dev \
    cython && \
  rm -rf /var/lib/apt/lists/* && \
  pip3 install twine && \
  pip3 install --upgrade keyrings.alt


