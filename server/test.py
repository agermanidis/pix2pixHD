### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
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

for i, data in enumerate(dataset):
    
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    print('label', data['label'])
    print('inst', data['inst'])
    minibatch = 1 
    generated = model.inference(data['label'], data['inst'])    
    print('generated', generated)    
        
    # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                        #    ('synthesized_image', util.tensor2im(generated.data[0]))])
    # img_path = data['path']
    # print('process image... %s' % img_path)
    # visualizer.save_images(webpage, visuals, img_path)
