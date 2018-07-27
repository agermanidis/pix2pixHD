# Pix2pix server
# ======================

import os
import json
import base64
import common
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from data.base_dataset import get_params, get_transform
import torch

# pix2pix options
opt = TestOptions().parse(save=False)
opt.nThreads = 1  
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.name = 'lateshow'
opt.netG = 'local'
opt.ngf = 32
opt.resize_or_crop = 'none'

model = create_model(opt)

# Server configs
PORT = 23100
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Take in base64 string and return PIL image
def stringToImage(base64_string):
  imgdata = base64.b64decode(base64_string)
  return Image.open(io.BytesIO(imgdata))

# Convert PIL Image to an RGB image(technically a numpy array) that's compatible with opencv
def toRGB(image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def main(input_img):
  image = stringToImage(input_img)
  label = toRGB(image)
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
  return base64.b64encode(buffer.getvalue())

# --- 
# Server Routes
# --- 
# Base route, functions a simple testing 
@app.route('/')
def index():
  return jsonify(status="200", message='pix2pixHD is running', query_route='/query', test_route='/test')

# Test the model with a fix to see if it's working
@app.route('/test')
def query():
  results = main(None)
  return jsonify(status="200", model='pix2pixHD', response=results)

# When a client socket connects
@socketio.on('connect', namespace='/query')
def new_connection():
  emit('successful_connection', {"data": "connection established"})

# When a client socket disconnects
@socketio.on('disconnect', namespace='/query')
def disconnect():
  print('Client Disconnect')

# When a client sends data. This should call the main() function
@socketio.on('update_request', namespace='/query')
def new_request(request):
  results = main(request["data"])
  emit('update_response', {"results": results})

if __name__ == '__main__':
  socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
