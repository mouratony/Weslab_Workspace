FROM condaforge/miniforge3

# Install system build tools and HDF5 development headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libhdf5-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your environment specification and create the environment
COPY environment.yml .
RUN conda env create -f environment.yml

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wesl_tony_env"]
CMD ["bash"]
