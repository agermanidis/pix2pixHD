# Run inference over one image
# ==============

import os
import cv2
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from data.base_dataset import get_params, get_transform
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

model = create_model(opt)
raw_img = cv2.imread('./imgs/lateshow_pose.jpg')
# img_resize = cv2.resize(raw_img, (512, 1024), interpolation = cv2.INTER_AREA)

label = Image.fromarray(raw_img)
params = get_params(opt, label.size)
transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
label_tensor = transform_label(label) * 255.0
inst_tensor = transform_label(label)
label_tensor = label_tensor.unsqueeze(0)
inst_tensor = inst_tensor.unsqueeze(0)
generated = model.inference(label_tensor, inst_tensor)
im = util.tensor2im(generated.data[0])
im_pil = Image.fromarray(im)
buffer = io.BytesIO()
im_pil.save(buffer, format='JPEG')
cv2.imwrite('./outputs/test.jpg', im_pil)
