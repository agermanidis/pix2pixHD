# Pytorch installation should be already in machine
# See: https://github.com/pytorch/pytorch#docker-image
FROM pytorch

# pix2pix Dependencies
RUN pip install dominate
RUN git clone https://github.com/NVIDIA/pix2pixHD
WORKDIR /workspace/pix2pixHD