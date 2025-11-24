FROM condaforge/miniforge3

# Install system build tools and HDF5 development headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libhdf5-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

SHELL ["bash", "-lc"]

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && \
    echo "conda activate wesl_tony_env" >> /root/.bashrc

CMD ["bash"]
