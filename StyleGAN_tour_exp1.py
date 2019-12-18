#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Let's go on a tour of the StyleGAN latent space.


# In[ ]:


# First things first...

import cv2
import random
import pandas as pd

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





# In[ ]:


# Load in latent vectors
vec1 = np.load('latent_representations/pat_flynn_01.npy')
vec2 = np.load('latent_representations/kevin_bowyer_01.npy')
vec3 = np.load('latent_representations/walter_scheirer_01.npy')
# vec4 = np.load('latent_representations/arnold_schwarzenegger_01.npy')


# In[ ]:


print(vec1.shape)


# In[ ]:


# Note for later: Is there a way to determine how well the encoder will encode the image originally fed in?


# In[ ]:


# zero_chick = generate_image(np.zeros(vec1.shape))
# zero_chick


# In[ ]:


# def change_identities(lat_vec1, lat_vec2, steps=60, fname='change_identities.avi'):
    
#     ### help ### 
#     ### https://gist.github.com/matpalm/23dc5804c6d673b800093d0d15e5de0e (author: mat kelcey) ###
 
#     lat_vec1_slim = np.reshape(lat_vec1, [1, lat_vec1.shape[0] * lat_vec1.shape[1]])
#     lat_vec2_slim = np.reshape(lat_vec2, [1, lat_vec2.shape[0] * lat_vec2.shape[1]])
    
#     assert(lat_vec1_slim.shape == lat_vec2_slim.shape, 'Latent vectors have different shape.')
    
#     z = np.empty((steps, lat_vec1_slim.shape[1]))   
#     for i, alpha in enumerate(np.linspace(start=1.0, stop=0.0, num=steps)):
#         z[i] = alpha * lat_vec1_slim + (1.0-alpha) * lat_vec2_slim
 
#     ### end help ###

#     images = []
#     for i in range(steps):
#         curr_vec = np.reshape(z[i], [18, 512]) # back to original shape
#         curr_img = generate_image(curr_vec)
#         images.append(np.array(curr_img))
        
#     s = images[0].shape
#     videowriter =  cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), steps/10, (s[1], s[0]))
#     for i in range(len(images)):
#         videowriter.write(images[i][...,::-1])
#     videowriter.release()


# In[ ]:


## From PF --> KB
# change_identities(vec1, vec2, steps=60, fname='pf_to_kb.avi')


# In[ ]:


## From 0 --> PF
# change_identities(np.zeros(vec1.shape), vec1, steps=90, fname='zero_to_pf.avi')


# In[ ]:


## From KB --> PF 
# change_identities(vec2, vec1, steps=60, fname='kb_to_pf.avi')


# In[ ]:


# SECOND HALF -- Which of the 18 vectors are identity salient?


# In[ ]:


# Let's mess around with KB...


# In[ ]:


def perturb_vector_axis(vector=vec1, num_images=60, axis=0, vid_fname='videos/test_video.avi'):

    # Get base image.
    base_img = generate_image(vector)
    # Get base image face encoding.
    base_face_encoding = face_recognition.face_encodings(np.array(base_img))[0]

    # Set up lists.    
    perts = []
    
    sg_distances = []
    fr_distances = []
    
    face_match = []

    # Set up video.
    s = np.array(base_img).shape
    videowriter =  cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'mp4v'), 5, (s[1], s[0]))
    
    for i in range(num_images):

        # Copy original vector
        vector_copy = np.copy(vector)

        # Perturb vector copy at specified axis.
        rnd = np.random.RandomState(i)
        pert = rnd.randn(Gs.input_shape[1])
        vector_copy[axis] = vector_copy[axis] + pert

        # Track vector perturbances for later analysis.
        perts.append(pert)

        # Track distance between vectors in sg latent space.
        sg_distances.append(np.linalg.norm(vector_copy - vector))
        # np.linalg.norm(vector_copy - vector) is Euclidean distance between base vector and perturbed vector.

        # Get perturbed image.
        perturbed_img = generate_image(vector_copy)
        # Get perturbed image face encoding.
        perturbed_face_encodings = face_recognition.face_encodings(np.array(perturbed_img))

        ### Found a face.
        if len(perturbed_face_encodings) > 0:
            perturbed_face_encoding = perturbed_face_encodings[0]
            fr_dist = face_recognition.face_distance([base_face_encoding], perturbed_face_encoding)[0]
            match = fr_dist < 0.4
            face_match.append(str(match)) # IS FOUND FACE CLOSE ENOUGH?

        ### Did NOT find a face.
        else:
            fr_dist = 1.0
            match = 'No Face Found'
            face_match.append(match) # NO FACE FOUND

        # Track distance between encodings in face_recognition space.
        fr_distances.append(fr_dist)
        
        # Write on image.
        draw = ImageDraw.Draw(perturbed_img)
        font = ImageFont.truetype('./fonts/GeosansLight.ttf', 40)
        col = 'green' if match==True else 'red'
        draw.text((40, 40), 'Face Match: '+str(match), font=font, fill=col)
        
        # Write image to video.
        videowriter.write(np.array(perturbed_img)[...,::-1])

    # Release handle.
    videowriter.release()
    
    # Save data in pandas df.
    df = pd.DataFrame([perts, sg_distances, fr_distances, face_match]).transpose()
    df.columns = ['perturbances', 'sg_distances', 'fr_distances', 'face_match']

    # To send back to user.
    return_dict = {}
    return_dict['df'] = df
    return_dict['vector'] = vector
    return_dict['axis'] = axis
    return_dict['base_image'] = base_img
    return_dict['base_face_encoding'] = base_face_encoding
    
    return return_dict


# # exp1
# 
# Let's make sure that the number of False matches in face_match decreases as axis increases. 
# 
# i.e. Let's see if the finer 'knobs' don't change as much in the face_recognition space.

# In[ ]:


# This is just for one vector (vec2 = KB) ... 
# We need to run this with more subjects obviously.

### PF

all_axes_dict = {}
for a in range(18):
    print('Running axis '+str(a)+'.')
    axis_dict = perturb_vector_axis(vector=vec1, axis=a, vid_fname='videos/exp1_vec1_axis'+str(a)+'.avi')
    all_axes_dict[a] = axis_dict


# In[ ]:


with open('all_axes_dict_vec1.pickle', 'wb') as f:
    pickle.dump(all_axes_dict, f)
    
### WALTER

all_axes_dict = {}
for a in range(18):
    print('Running axis '+str(a)+'.')
    axis_dict = perturb_vector_axis(vector=vec3, axis=a, vid_fname='videos/exp1_vec3_axis'+str(a)+'.avi')
    all_axes_dict[a] = axis_dict


# In[ ]:


with open('all_axes_dict_vec3.pickle', 'wb') as f:
    pickle.dump(all_axes_dict, f)

