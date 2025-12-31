FROM nvidia/cudagl:11.3.1-base-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


FROM nvidia/cudagl:11.3.1-base-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ===== ユーザ設定: ホストの UID/GID に合わせた cudagl を作成 =====
ARG UID=1000
ARG GID=1000
ENV USERNAME=cudagl

# 非対話モード & タイムゾーン
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN set -eux; \
    # タイムゾーン設定（tzdata が入るのは後でもよいですが、入っていればこれで OK）
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone || true; \
    \
    # グループ作成/更新
    if getent group "${USERNAME}" >/dev/null 2>&1; then \
        groupmod -o -g "${GID}" "${USERNAME}"; \
    else \
        groupadd -g "${GID}" "${USERNAME}"; \
    fi; \
    \
    # ユーザ作成/更新（uid/gid をホストに合わせる）
    if id "${USERNAME}" >/dev/null 2>&1; then \
        usermod -o -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    else \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    fi; \
    \
    mkdir -p /home/${USERNAME}; \
    chown -R "${UID}:${GID}" /home/${USERNAME}



# # Optimize the apt mirror for faster package downloads
# RUN sed -i.bak 's|http://archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|' /etc/apt/sources.list

# # Set up timezone configuration to avoid interactive prompts
# RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
#     tzdata && \
#     ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#     dpkg-reconfigure --frontend noninteractive tzdata && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo less tmux bash-completion curl wget coreutils git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenGL-related libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev freeglut3-dev mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install archive tools (xz-utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xz-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install build tools and dependencies for Python build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential zlib1g-dev libffi-dev libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up temporary XDG_RUNTIME_DIR
RUN mkdir -p /tmp/xdg_runtime_dir && chmod 777 /tmp/xdg_runtime_dir
RUN echo "export XDG_RUNTIME_DIR=/tmp/xdg_runtime_dir" >> ~/.bashrc


#####################################################
# 		   	     Python 3.10.12
#####################################################
# Install dependencies for Tkinter
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-dev libxext-dev libxft-dev libxss-dev \
    libxrandr-dev tk-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install SQLite development libraries (required for _sqlite3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.10.12 with Tkinter support
RUN curl -O https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz \
    && tar -xf Python-3.10.12.tar.xz \
    && cd Python-3.10.12 \
    && ./configure --enable-optimizations --with-tcltk \
    && make -j8 \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.10.12 Python-3.10.12.tar.xz

# Set up symbolic links for python and pip
RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python
RUN ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install Tkinter for Python GUI applications
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC apt-get install -y --no-install-recommends \
    python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#####################################################
# 		   		   Mujoco 3.2.3
#####################################################
RUN apt-get update \
    && apt-get install -y manpages-dev psmisc tree\
    # Install gl/x package for visualization inside docker
    && apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 libzmq3-dev
RUN pip install mujoco==3.2.3

#####################################################
# 		   		  ipdb dependencies
#####################################################
RUN apt update
RUN apt install -y libreadline-dev libncursesw5-dev \
    libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev \
    libffi-dev

RUN apt update && sudo apt install -y libncurses-dev libreadline-dev


#####################################################
# 		   		  Webcamera
#####################################################
RUN apt-get update && apt-get install -y v4l-utils && rm -rf /var/lib/apt/lists/*



# #####################################################
# # Orbbec SDK
# #####################################################
# # !!! ホスト側で，udevのルールを作成しておくこと !!!
# # UDEV 設定, リポDL->udev.sh実行
# # https://github.com/orbbec/pyorbbecsdk/tree/v2-main
# # dockerではudev前提で，orbbecsdkとpyorbbecsdkをインストール
# # SDKランタイムのインストール
# RUN apt-get update && apt-get install -y udev wget && \
#     wget -O /tmp/OrbbecSDK.deb https://github.com/orbbec/OrbbecSDK_v2/releases/download/v2.5.5/OrbbecSDK_v2.5.5_amd64.deb && \
#     dpkg -i /tmp/OrbbecSDK.deb || apt-get -y -f install && \
#     rm -f /tmp/OrbbecSDK.deb
# # Pythonバインディングのインストール
# RUN mkdir -p /home/${USERNAME}/workspace/orbbec_sdk && \
#     wget -O /home/${USERNAME}/workspace/orbbec_sdk/pyorbbecsdk-2.0.13-cp39-cp39-linux_x86_64.whl \
#         https://github.com/orbbec/pyorbbecsdk/releases/download/v2.0.13/pyorbbecsdk-2.0.13-cp39-cp39-linux_x86_64.whl && \
#     python -m pip install --no-cache-dir /home/${USERNAME}/workspace/orbbec_sdk/pyorbbecsdk-2.0.13-cp39-cp39-linux_x86_64.whl
# # ライブラリパスを通す
# ENV LD_LIBRARY_PATH=/opt/orbbec/OrbbecSDK/lib:${LD_LIBRARY_PATH}


#####################################################
# 		   		   USER CHANGE
#####################################################
# 非 root ユーザ（ホストと同じ UID/GID の cudagl）で動かす
USER ${USERNAME}
WORKDIR /home/${USERNAME}

CMD ["/bin/bash"]
