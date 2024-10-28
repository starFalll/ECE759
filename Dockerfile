# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMP (comes with GCC)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Eigen
RUN mkdir /eigen && \
    cd /eigen && \
    wget -O eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xzf eigen.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && cd build && \
    cmake .. && make install && \
    rm -rf /eigen

RUN cd /usr/local/include/ && \
    ln -sf eigen3/Eigen Eigen

# Set working directory
WORKDIR /workspace

# Command to keep the container running
CMD ["bash"]

