FROM tensorflow/tensorflow:latest-gpu

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install wget

RUN apt-get install build-essential
RUN apt-get update && apt-get install -y cmake
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get install -y pkg-config
RUN apt-get update && apt-get install -y libavcodec-dev
RUN apt-get update && apt-get install -y libavformat-dev
RUN apt-get update && apt-get install -y libswscale-dev
RUN apt-get update && apt-get install -y python-opencv


COPY . /app
WORKDIR /app
