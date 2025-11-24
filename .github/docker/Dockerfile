FROM ubuntu:24.04

# Install Python, venv, pip, git, build tools
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install packages
RUN pip install --upgrade pip && \
    pip install sphinx sphinx-rtd-theme "jax[cpu]"

# Set working directory
WORKDIR /workspace
