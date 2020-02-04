# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
import os
import boto3
import json
import tempfile
import urllib2

import mxnet as mx
import numpy as np
from datetime import datetime

from PIL import Image
from collections import namedtuple

app = Flask(__name__)

Batch = namedtuple('Batch', ['data'])

f_params = 'squeezenet_v1.0-0000.params'
f_symbol = 'squeezenet_v1.0-symbol.json'

# bucket = 'smallya-test'
# s3 = boto3.resource('s3')
# s3_client = boto3.client('s3')
#
# #params
# f_params_file = tempfile.NamedTemporaryFile()
# s3_client.download_file(bucket, f_params, f_params_file.name)
# f_params_file.flush()
#
# #symbol
# f_symbol_file = tempfile.NamedTemporaryFile()
# s3_client.download_file(bucket, f_symbol, f_symbol_file.name)
# f_symbol_file.flush()

f_params_file = f_params
f_symbol_file = f_symbol

def load_model(s_fname, p_fname):
    """
    Load model checkpoint from file.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    symbol = mx.symbol.load(s_fname)
    save_dict = mx.nd.load(p_fname)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params

def predict(url, mod, synsets):
    '''
    predict labels for a given image
    '''

    # req = urllib2.urlopen(url)
    # img_file = tempfile.NamedTemporaryFile()
    # img_file.write(req.read())
    # img_file.flush()
    #
    # img = Image.open(img_file.name)

    img = Image.open("dogs_small.jpg")

    # PIL conversion
    #size = 224, 224
    #img = img.resize((224, 224), Image.ANTIALIAS)

    # center crop and resize
    # ** width, height must be greater than new_width, new_height
    new_width, new_height = 224, 224
    width, height = img.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img = img.crop((left, top, right, bottom))
    # convert to numpy.ndarray
    sample = np.asarray(img)
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    img = np.swapaxes(sample, 1, 2)
    img = img[np.newaxis, :]

    # forward pass through the network
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    out = ''
    for i in a[0:5]:
        out += 'probability=%f, class=%s , ' %(prob[i], synsets[i])
    out += "\n"
    return out

with open('synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

def lambda_handler():

    url = ''
    # try:
    #     # API Gateway GET method
    #     if event['httpMethod'] == 'GET':
    #         url = event['queryStringParameters']['url']
    #     # API Gateway POST method
    #     elif event['httpMethod'] == 'POST':
    #         data = json.loads(event['body'])
    #         url = data['url']
    # except KeyError:
    #     # direct invocation
    #     url = event['url']

    t1 = datetime.now()
    sym, arg_params, aux_params = load_model(f_symbol_file, f_params_file)
    mod = mx.mod.Module(symbol=sym, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    t2 = datetime.now()
    loadmodeldelta = t2 - t1
    t1 = datetime.now()
    labels = predict(url, mod, synsets)
    t2 = datetime.now()
    delta = t2 - t1

    out = {
            "headers": {
                "content-type": "application/json",
                "Access-Control-Allow-Origin": "*"
                },
            "body": labels,
            "predicttime": str(delta.total_seconds()),
            "loadmodeldelta": str(loadmodeldelta.total_seconds()),
            "model": "squeezenet_v1.0",
            "statusCode": 200
          }
    return out

count = 0
@app.route('/HealthCheck')
def health_check():
    global count
    if count >= 10:
        return 'Not OK', 404
    else:
        return 'OK'


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    global count
    count = count + 1
    ret = lambda_handler()
    count = count - 1
    return ret


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
