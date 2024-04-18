# Use Miniconda base image compatible with ARM64
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Update Conda and Pip, and add necessary channels
RUN conda update -n base -c defaults conda && \
    conda install pip && \
    pip install --upgrade pip && \
    conda config --add channels conda-forge

# Install Python and basic packages with Conda
RUN conda create -n EML python=3.10 scipy numpy matplotlib

# Activate the environment and install PyTorch packages via Pip
SHELL ["conda", "run", "-n", "EML", "/bin/bash", "-c"]

# Using pip to install PyTorch packages directly from PyTorch's wheel links
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Set the default command to execute
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "EML", "python"]
