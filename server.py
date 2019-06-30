from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
#from urllib.request import urlretrieve
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import json
from Eval import *
import torch
import math

import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt

import tensorflow as tf

irange = range


_IMG_WIDTH_ = 0
_IMG_HEIGHT_ = 0

def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):

    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename, format='JPEG')


def makeBinary(mask, main):
    binary = np.array((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == main:
                binary[i][j][0] = 1.
                binary[i][j][1] = 1.
                binary[i][j][2] = 1.
            else:
                binary[i][j][0] = 0.0
                binary[i][j][1] = 0.0
                binary[i][j][2] = 0.0

    return binary


_NN_INPUT_SIZE_ = (640, 640)

from PIL import Image, ImageDraw
def imageFromBase64(string):
    data = base64.b64decode(string)
    stream = BytesIO(data)
    img = Image.open(stream).convert('RGB')

    #draw = ImageDraw.Draw(img)
    #if not stream.closed:
    #    stream.close()
    return img

def getImageFromURL(url):
    req = requests.get(url)
    buffer = BytesIO(req.content)
    b64cover = base64.b64encode(buffer.getvalue())
    img = Image.open(buffer).convert('RGB')

    #if not buffer.closed:
    #    buffer.close()
    return img, str(b64cover, 'utf-8')

def imagePreprocessing(img):
    '''
    pads and resizes image to make square
    :param img: PIL Image
    :return: PIL Image
    '''
    arr = np.array(img)
    max_edge = max(arr.shape[0], arr.shape[1])
    old_width = arr.shape[1]
    old_height = arr.shape[0]
    dw = (max_edge - arr.shape[1]) // 2
    dh = (max_edge - arr.shape[0]) // 2
    padded = np.pad(arr, ((dw, dw), (dh, dh), (0, 0)), mode='constant')

    return Image.fromarray(padded).resize(_NN_INPUT_SIZE_), dw / arr.shape[1], dh / arr.shape[0], old_width, old_height

@Request.application
def application(request):
    s = request.data.decode('utf-8').replace("'", '"')
    #json_res = ''
    if request.method == 'POST':
        json_dict = json.loads(s)
        print('Length', len(json_dict['user_img']))
        b64img = json_dict['user_img'].split(',')[1]

        #imageFromBase64(b64img)
        content_image, dw, dh, oldw, oldh = imagePreprocessing(imageFromBase64(b64img))
        print('Content image:', content_image.size)
        content_image.save('content.jpg')
        #getImageFromURL(json_dict['search_img'])
        cover, coverb64 = getImageFromURL(json_dict['search_img'])
        style_image, _, _, _, _ = imagePreprocessing(cover)
        print('Style image:', style_image.size)
        style_image.save('style.jpg')
        #level = int(json_dict['level'])
        level = 5
        transfer_style_portrait(content_image, style_image, level+4, level/2)
        buffer = BytesIO()
        #img_bytes = buffer.getvalue()
        back_img = Image.open('cumming0.jpg')
        face_img = Image.open('cumming1.jpg')
        #abspath = os.path.dirname(os.path.abspath(__file__))
        #abspath = os.path.join(abspath, 'content.jpg')
        #resized, seg_map = run_visualization(abspath)
        #print(np.unique(seg_map))

        #binary_map = (seg_map / seg_map.max()).astype('int32')
        #binay_img = Image.fromarray(binary_map).resize(back_img.size)
        #binary_map = np.array(binay_img)
        #binary_map = np.array([binary_map, binary_map, binary_map])
        #binary_map = np.swapaxes(binary_map, 0, 1)
        #binary_map = np.swapaxes(binary_map, 1, 2)

        #binary_map_face = binary_map
        #binary_map_back = 1 - binary_map_face

        #res_img = (binary_map_back * np.array(back_img) + binary_map_face * np.array(face_img))
        #res_img = Image.fromarray(res_img.astype('uint8'))
        face_img = np.array(face_img)
        back_img = np.array(back_img)
        res_img = np.zeros(face_img.shape, dtype='uint8')
        for i in range(res_img.shape[0]):
            for j in range(res_img.shape[1]):
                if i > res_img.shape[0] // 4 and i < 3 * res_img.shape[0] / 4:
                    if j > res_img.shape[1] // 4 and j < 3 * res_img.shape[1] / 4:
                        res_img[i][j] = back_img[i][j]#(face_img[i][j] + back_img[i][j])
                        continue
                res_img[i][j] = back_img[i][j]
        res_img = Image.fromarray(res_img)
        dw = int(dw * res_img.size[1])
        dh = int(dh * res_img.size[0])
        dd = max(dw, dh)
        box = (dd, dd, res_img.size[1] - dd, res_img.size[1] - dd)

        res_img = res_img.crop(box)
        res_img = res_img.resize((oldw, oldh))
        res_img.save(buffer, format='JPEG')
        res_img.save('fcking_shit.jpg')
        img_bytes = buffer.getvalue()

        b64img = base64.b64encode(img_bytes)
        res = dict()
        res['img'] = 'data:image/jpeg;base64,' + str(b64img, 'utf-8')
        res['cover'] = 'data:image/png;base64,' + coverb64
        res['type'] = 'ok'

        # url = "https://api.imgur.com/3/image"
        # querystring = {"access_token": "59aa2669edd3ed0a66eefa9cd8a794f690620a7e", "client_id": "018ebfa932f27b1"}
        # payload = res['img'].split(',')[1]
        # headers = {
        #     'cookie': "UPSERVERID=upload.i-0224b52b6548410d3.production; IMGURSESSION=bc8a9d333b1802bbd4f99c5a1bfb5ac1; _nc=1",
        #     'content-type': "image/jpeg"
        # }
        # response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
        # imgur = response.json()
        # print(imgur)
        # res['link'] = imgur['data']['link']

        json_res = json.dumps(res)
    elif request.method == 'OPTIONS':
        json_res = json.dumps({'type': 'ok'})
    else:
        json_res = json.dumps({'type': 'unexpected error'})

    resp = Response(bytes(json_res, 'utf-8'), mimetype='application/json')
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.headers.add('Access-Control-Allow-Methods', 'POST')
    resp.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept')


    return resp




if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    #application({'method': 'POST'})
    run_simple('localhost', 1952, application)