#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# First things first...

import cv2
import random
import pandas as pd
import statistics

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Taken from pretrained_example.py
import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator

# Off-the-shelf recognizer
import face_recognition

# Plotting
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image, ImageDraw, ImageFont


# In[ ]:


# Plot latent vectors of shape 18x512
def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((1024, 1024))


# In[ ]:


def setup():
    tflib.init_tf()
    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
    generator = Generator(Gs, batch_size=1, randomize_noise=False) # -- RUNNING >1 TIMES THROWS ERROR
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    return [_G, _D, Gs, generator, fmt]


# In[ ]:


# Only run once.
[_G, _D, Gs, generator, fmt] = setup()


# In[ ]:


# # Load in latent vectors
# vec1 = np.load('latent_representations/04229d87_01.npy')
# vec2 = np.load('latent_representations/04551d40_01.npy')


# In[ ]:


def interpolate(dirname1, dirname2, steps=60):

    # Load and convert latent vectors.
    vec1 = np.load('./data2/'+dirname1+'/'+dirname1+'_01.npy')
    vec2 = np.load('./data2/'+dirname2+'/'+dirname2+'_01.npy')

    vec1_slim = np.reshape(vec1, [1, vec1.shape[0] * vec1.shape[1]])
    vec2_slim = np.reshape(vec2, [1, vec2.shape[0] * vec2.shape[1]])
    
#     print(vec1_slim.shape, vec2_slim.shape)
    
    # Load and learn encodings from raw images.  
    img1 = face_recognition.load_image_file('./data2/'+dirname1+'/'+dirname1+'.JPG')
    img2 = face_recognition.load_image_file('./data2/'+dirname2+'/'+dirname2+'.JPG')
    
    enc1 = face_recognition.face_encodings(img1)[0]
    enc2 = face_recognition.face_encodings(img2)[0]
    
    known_encodings = [enc1, enc2]
    
#     print(len(known_encodings))
    
    z = np.empty((steps, vec1_slim.shape[1]))   
    for i, alpha in enumerate(np.linspace(start=1.0, stop=0.0, num=steps)):
        z[i] = (1.0-alpha) * vec1_slim + alpha * vec2_slim

#     print(z.shape)
        
    images = []
    distances = []
    
    for i in range(steps):

        curr_vec = np.reshape(z[i], [18, 512]) # back to original shape
        curr_img = generate_image(curr_vec)
        
        curr_enc = face_recognition.face_encodings(np.array(curr_img))[0]
        distance = face_recognition.face_distance(known_encodings, curr_enc)
        
        images.append(np.array(curr_img))
        distances.append(distance)
        
    return_dict = {
        'images': images,
        'distances': distances
    }
    
    return return_dict


# In[ ]:


return_dict = interpolate('02463d214', '04201d96')
with open('return_dict.pkl', 'wb') as f:
	pickle.dump(return_dict, f)


# In[ ]:


# images = return_dict['images']


# In[ ]:


# s = images[0].shape
# steps = 60
# videowriter =  cv2.VideoWriter('interpolate_test.avi', cv2.VideoWriter_fourcc(*'mp4v'), steps/10, (s[1], s[0]))
# for i in range(len(images)):
#     videowriter.write(images[i][...,::-1])
# videowriter.release()

