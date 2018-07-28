# Run inference over a single image
# =================
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

# Model Options that match the training
opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.name = 'lateshow'
#opt.netG = 'local'
#opt.ngf = 32
opt.resize_or_crop = 'none'
opt.use_features = False
opt.no_instance = True
opt.label_nc = 0

# Load the model
model = create_model(opt)

# Load a hard code an image, just to test
raw_img = Image.open("imgs/lateshow_pose.jpg")

# Prepare image
params = get_params(opt, raw_img.size)
transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
label_tensor = transform_label(raw_img)
label_tensor = label_tensor.unsqueeze(0)

# Get fake image
generated = model.inference(label_tensor, None)

# Save img
im = util.tensor2im(generated.data[0])
im_pil = Image.fromarray(im)
im_pil.save('outputs/test.jpg')

