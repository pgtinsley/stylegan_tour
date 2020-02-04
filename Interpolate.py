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
# import matplotlib.pyplot as plt
# %matplotlib inline

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


def interpolate(root1, root2, steps=60):

    # Load and convert latent vectors.
    vec1 = np.load('./latent_representations/'+root1+'_01.npy')
    vec2 = np.load('./latent_representations/'+root2+'_01.npy')

    vec1_slim = np.reshape(vec1, [1, vec1.shape[0] * vec1.shape[1]])
    vec2_slim = np.reshape(vec2, [1, vec2.shape[0] * vec2.shape[1]])
    
    # print(vec1_slim.shape, vec2_slim.shape)
    
    # Load and learn encodings from raw images.  
    img1 = face_recognition.load_image_file('./aligned_images/'+root1+'_01.png')
    img2 = face_recognition.load_image_file('./aligned_images/'+root2+'_01.png')
    
    #enc1 = face_recognition.face_encodings(img1)[0]
    #enc2 = face_recognition.face_encodings(img2)[0]
    enc1 = np.random.random(128)
    enc2 = np.random.random(128)
    
    known_encodings = [enc1, enc2]
    
    images = []
    fr_distances = []
    axis_means = []
    
    z = np.empty((steps, vec1_slim.shape[1]))   
    for i, alpha in enumerate(np.linspace(start=1.0, stop=0.0, num=steps)):
        
        z[i] = (alpha) * vec1_slim + (1.0-alpha) * vec2_slim
        
        curr_vec = np.reshape(z[i], [18, 512]) # back to original shape
        curr_img = np.array(generate_image(curr_vec))
        
        #curr_enc = face_recognition.face_encodings(curr_img)[0]
        curr_enc = np.random.random(128)
        fr_distance = face_recognition.face_distance(known_encodings, curr_enc)
        
        images.append(np.array(curr_img))
        fr_distances.append(fr_distance)
        axis_means.append(np.mean(curr_vec, axis=1))
    
    return_dict = {
        'images': images,
        'fr_distances': fr_distances,
        'axis_means': axis_means,
    }
    
    return return_dict


# In[ ]:


return_dict = interpolate('04557d14_hr', '04558d30_hr')


# In[ ]:


# Same-Subject
# Get 3 combinations: hr1->hr2, lr1->lr2, hr1->lr1
combos = [
    ('04261d50_hr','04261d51_hr'),
    ('04261d53_lr', '04261d52_lr'),
    ('04261d50_hr', '04261d53_lr')
]
# 04261d53_lr_01.png # bad
# 04261d52_lr_01.png
# 04261d51_hr_01.png
# 04261d50_hr_01.png
for i, combo in enumerate(combos):
    return_dict = interpolate(combo[0], combo[1])
    # Save pkl
    with open('./return_dict_ss_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(return_dict, f)
    # Save video
    images = return_dict['images']
    s = images[0].shape
    steps = 60
    videowriter =  cv2.VideoWriter('./video_ss_'+str(i)+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), steps/10, (s[1], s[0]))
    for i in range(len(images)):
        videowriter.write(images[i][...,::-1])
    videowriter.release()


# In[ ]:


# Inter-Subject
# Get 3 combinations: hr1->hr2, lr1->lr2, hr1->lr2
combos = [
    ('04261d50_hr', '02463d214_hr'),
    ('04261d53_lr', '02463d217_lr'),
    ('04261d53_lr', '02463d214_hr')
]
for i, combo in enumerate(combos):
    return_dict = interpolate(combo[0], combo[1])
    # Save pkl
    with open('./return_dict_is_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(return_dict, f)
    # Save video
    images = return_dict['images']
    s = images[0].shape
    steps = 60
    videowriter =  cv2.VideoWriter('./video_is_'+str(i)+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), steps/10, (s[1], s[0]))
    for i in range(len(images)):
        videowriter.write(images[i][...,::-1])
    videowriter.release()


# In[ ]:


## Interesting idea... what about if we compare the results of:
# generated_image1 --> slide through 9216 SG feature space --> generated_image2
# aligned_image1 --> slide through 1024x1024 feature space --> aligned_image2
# generated_image1_slim --> slide through <9216 SG-slim feature space --> generated_image2_slim

