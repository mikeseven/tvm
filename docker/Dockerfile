FROM simaai/ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install \
    build-essential ccache \
    wget curl awscli cmake unzip git && \
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

RUN apt-get update && \
  apt-get install -y \
    libopenblas-dev \
    antlr4 \
    cython && \
  rm -rf /var/lib/apt/lists/* && \
  pip3 install twine pyyaml && \
  pip3 install --upgrade keyrings.alt


