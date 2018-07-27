nvidia-docker run --rm -it -v /home/paperspace/pix2pixHD:/workspace/pix2pixHD \
  -v /storage:/workspace/storage \
  --shm-size 16G \
  pix2pixhd:latest bash
