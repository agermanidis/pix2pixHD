nvidia-docker run --rm -it -v /home/paperspace/pix2pixHD:/workspace/pix2pixHD \
  -v /storage:/workspace/storage \
  -p 23100:23100 \
  --shm-size 16G \
  pix2pixhdserver:latest bash
