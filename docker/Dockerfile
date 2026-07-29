# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  gcc \
  curl \
  wget \
  ffmpeg \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgl1 \
  libgl-dev \
  libegl1 \
  libegl-dev \
  libglx0 \
  libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv/)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create virtual environment with uv
WORKDIR /boxer
RUN uv venv /boxer/venv --python python3.12 --seed
ENV VIRTUAL_ENV=/boxer/venv
ENV PATH="/boxer/venv/bin:${PATH}"

RUN uv pip install \
  'torch>=2.0' \
  numpy \
  opencv-python \
  tqdm \
  dill \
  Pillow

# To support Project Aria loading
RUN uv pip install projectaria-tools || true

# 3D interactive viewer for view_*.py scripts
RUN uv pip install \
  moderngl \
  moderngl-window \
  imgui-bundle

# test deps
RUN uv pip install pytest pytest-cov

COPY . /boxer
RUN chmod +x /boxer/docker/entrypoint.sh

RUN echo 'source /boxer/venv/bin/activate' >> /root/.bashrc

ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

ENTRYPOINT ["/boxer/docker/entrypoint.sh"]
CMD ["python", "run_boxer.py", "--help"]
