# Pytorch installation should be already in machine
# See: https://github.com/pytorch/pytorch#docker-image
FROM pytorch

# pix2pix Dependencies
RUN pip install dominate
RUN git clone https://github.com/NVIDIA/pix2pixHD

# Install Server Dependencies
RUN pip3 install cython common flask flask_cors flask_socketio pillow gevent

WORKDIR /workspace/pix2pixHD