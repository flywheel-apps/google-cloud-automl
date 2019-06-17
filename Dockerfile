FROM alpine:3.8
MAINTAINER Flywheel <support@flywheel.io>

ENV FLYWHEEL /flywheel/v0
WORKDIR ${FLYWHEEL}
RUN set -eux \
    && apk add --no-cache \
        bash \
        git \
        build-base \
        freetype-dev \
        jpeg-dev \
        libffi-dev \
        libpng-dev \
        openssl-dev \
        py3-pip \
        python3-dev \
    && rm -rf /var/cache/apk/* \
    && pip3 install --upgrade pip \
    && mkdir -p ${FLYWHEEL}/input ${FLYWHEEL}/output

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
