FROM ubuntu:16.04


RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    python-pip \
    python2.7 \
    python2.7-dev \
    ssh \
    python-tk \
    libgtk2.0-dev -y \
    && apt-get autoremove \
    && apt-get clean

COPY requirements.txt /tmp
WORKDIR /tmp

RUN pip install --upgrade pip
RUN pip install -r requirements.txt