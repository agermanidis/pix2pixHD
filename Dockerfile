# Pytorch installation should be already in machine
# See: https://github.com/pytorch/pytorch#docker-image
FROM pytorch

RUN apt-get update && apt-get install -y libgtk2.0-dev && apt-get install -y nano
# pix2pix Dependencies
RUN pip install dominate
RUN git clone https://github.com/NVIDIA/pix2pixHD

# Install Server Dependencies
RUN pip install cython common flask flask_cors flask_socketio pillow gevent
RUN pip install opencv-python
RUN pip install torch torchvision

WORKDIR /workspace/pix2pixHD
