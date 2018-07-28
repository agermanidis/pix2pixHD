# Run inference over a multiple image
# =================
import os
from PIL import Image
import argparse
import cv2
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
import torch
from data.base_dataset import get_params, get_transform

class CustomOptions(TestOptions):
  def initialize(self):
   TestOptions.initialize(self)
   self.parser.add_argument('-imgs_dir', '--images_dir', dest='images_dir', type=str, default='.', help='Path of images')
   self.parser.add_argument('-img_f', '--images_format', dest='images_format', type=str, default='jpg', help='Format of')
   self.parser.add_argument('-vid_f', '--video_format', dest='video_format', type=str, default='mp4', help='Format of v')
   self.parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default='./outputs', help='Output direct')
   self.parser.add_argument('-v_name', '--video_name', dest='video_name', type=str, default='output', help='Video name')
   self.parser.add_argument('-v_f', '--video_format', dest='video_format', type=str, default='.mp4', help='Video Format')
   self.parser.add_argument('-make_video', '--make_video', dest='make_video', type=bool, default=True, help='Toggle video making')

# Model Options that match the training
opt = CustomOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.name = 'lateshow'
opt.resize_or_crop = 'none'
opt.use_features = False
opt.no_instance = True
opt.label_nc = 0

# Load the model
model = create_model(opt)

def main():
  print('Running pix2pixHD over all images in %s' % opt.images_dir)
  images = [img for img in os.listdir(opt.images_dir) if img.endswith(".%s" % opt.images_format)]

  for image in images:
    raw_img = Image.open(os.path.join(opt.images_dir, image))
    params = get_params(opt, raw_img.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(raw_img)
    label_tensor = label_tensor.unsqueeze(0)
    # Get fake image
    generated = model.inference(label_tensor, None)
    # Save img
    im = util.tensor2im(generated.data[0])
    im_pil = Image.fromarray(im)
    im_pil.save(os.path.join(opt.output_dir, image))

  if opt.make_video:
    print('Making video with %s' % opt.output_dir)
    images = [img for img in os.listdir(opt.output_dir) if img.endswith(".%s" % opt.images_format)]
    frame = cv2.imread(os.path.join(args.output_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(opt.output_dir, opt.video_name + "." + opt.video_format), -1, args.fps, (width,height))

    for image in images:
      video.write(cv2.imread(os.path.join(opt.output_dir, image)))

    cv2.destroyAllWindows()
    video.release()
    print('Done! Video ready.')

if __name__ == '__main__':
  main()

