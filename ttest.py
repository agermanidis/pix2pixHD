### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from PIL import Image
import cv2
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from data.base_dataset import get_params, get_transform
from torchvision import models, transforms

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.name = 'lateshow'
#opt.netG = 'local'
#opt.ngf = 32
opt.resize_or_crop = 'none'
opt.use_features = False
opt.no_instance = True
opt.label_nc = 0

model = create_model(opt)
raw_img = Image.open("imgs/lateshow_pose.jpg")

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

params = get_params(opt, raw_img.size)
transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
label_tensor = transform_label(raw_img)
label_tensor = label_tensor.unsqueeze(0)

#img_tensor = preprocess(raw_img)
#img_tensor = img_tensor.unsqueeze_(0)

generated = model.inference(label_tensor, None)
im = util.tensor2im(generated.data[0])
im_pil = Image.fromarray(im)
im_pil.save('outputs/test.jpg')


#for i, data in enumerate(dataset):
 #   if i >= 1:
  #      break
   # if opt.data_type == 16:
    #    print('data type is', opt.data_type, '16')
     #   data['label'] = data['label'].half()
      #  data['inst']  = data['inst'].half()
    #elif opt.data_type == 8:
     #   print('data type is', opt.data_type, '8')
      #  data['label'] = data['label'].uint8()
      #  data['inst']  = data['inst'].uint8()

    # print('data label', data['label'])
    #visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
#                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    #img_path = data['path']
    #print('process image... %s' % img_path)
    #visualizer.save_images(webpage, visuals, img_path)

