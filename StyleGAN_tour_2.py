#!/usr/bin/env python
# coding: utf-8

# ### Let's go on a tour of the StyleGAN latent space.

# In[1]:


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


# In[2]:


# Plot latent vectors of shape 18x512
def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((1024, 1024))


# In[3]:


def setup():
    tflib.init_tf()
    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
    generator = Generator(Gs, batch_size=1, randomize_noise=False) # -- RUNNING >1 TIMES THROWS ERROR
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    return [_G, _D, Gs, generator, fmt]


# In[4]:


# Only run once.
[_G, _D, Gs, generator, fmt] = setup()


# In[ ]:





# In[ ]:


# Load in latent vectors
# vec = np.load('latent_representations/kevin_bowyer_01.npy')
# vec = np.load('latent_representations/pat_flynn_01.npy')
# vec = np.load('latent_representations/walter_scheirer_01.npy')
# vec4 = np.load('latent_representations/arnold_schwarzenegger_01.npy')


# In[ ]:


# vec.shape


# In[5]:


def mc_perturb(base_vector, axis, pkl_fname, magnitudes=[0.05, 0.1, 0.5, 1, 5]):
    
    # Get base image.
    base_image = generate_image(base_vector)
    
    # Get base image fr encoding.
    base_image_encoding = face_recognition.face_encodings(np.array(base_image))[0]

    # Copy base vector.
    new_vector = np.copy(base_vector)

    return_dict = {}
    return_dict['base_vector'] = base_vector
    return_dict['axis'] = axis
    return_dict['magnitudes'] = magnitudes
    
    # Loop over magnitudes.
    for magnitude in magnitudes:

        return_dict[str(magnitude)] = {}

        # images = []
        sg_distances = []
        fr_distances = []
        
        # Loop over random seeds.
        for i in range(100):
                
            # Assign random state.
            rnd = np.random.RandomState(i)
            
            # Perturb specified axis in new vector.
            new_vector[axis] = new_vector[axis] + magnitude * rnd.randn(Gs.input_shape[1])

            # Save distance in SG space.
            sg_distances.append(np.linalg.norm(base_vector-new_vector))
            
            # Get new image.
            new_image = generate_image(new_vector)
            
            # # Save image for later.
            # images.append(new_image)
            
            # Get new image fr encoding.
            new_image_encodings = face_recognition.face_encodings(np.array(new_image))

            # FACE DETECTED/ENCODED
            if len(new_image_encodings) > 0:
                new_image_encoding = new_image_encodings[0]
                fr_distance = face_recognition.face_distance([base_image_encoding], new_image_encoding)[0]

            # NO FACE DETECTED
            else:
                fr_distance = 1.0

            # Save distance in FR space.
            fr_distances.append(fr_distance)
    
        # return_dict[str(magnitude)]['images'] = images
        return_dict[str(magnitude)]['sg_distances'] = sg_distances
        return_dict[str(magnitude)]['fr_distances'] = fr_distances        
    
    with open('axis{}.avi'.format(axis), 'wb') as f:
        pickle.dump(return_dict, f)
    
#     return return_dict


# In[11]:


dirnames = os.listdir('../data/FRGC/FRGC-2.0-dist/nd1/custom_100/')


# In[12]:


len(dirnames)


# In[13]:


dirnames[0:5]


# In[15]:


for dirname in dirnames:
    
    vec = np.load('../data/FRGC/FRGC-2.0-dist/nd1/custom_100/' + dirname + '/' + dirname + '_01.npy')
    
    for a in range(18):
        p_fname = '../data/FRGC/FRGC-2.0-dist/nd1/custom_100/' + dirname + '/' + dirname + '_axis' + str(a) + '.pkl'
        
        if not os.path.exists(p_fname):
            mc_perturb(vec, axis=a, pkl_fname=p_fname)
        else:
            print(p_fname + ' already exists')


# In[ ]:




