# -*- coding: utf-8 -*-

import os
os.system('pip install -r requirements.txt')
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import Augmentor

def get_pixel(image, i, j):
    """ Returns a pixel at coordinate (`i`, `j`). """
    return image.getpixel((i,j))

def create_augmentor_pipeline(dir_path):
    """ Creates a pipeline for generating extra images from images at folder `dir_path`."""
    p = Augmentor.Pipeline(dir_path)
    p.resize(probability=1,width=64,height=64)
    p.rotate90(probability=0.1)
    p.rotate(probability=0.2, max_left_rotation=5, max_right_rotation=10)
    p.skew_left_right(probability=0.1)
    p.greyscale(probability=1)
    return p


def load_data(dir_path):
    """ Loads all the images from directory `dir_path`, converts them to matrices and return a list."""
    files = os.listdir(dir_path)
    num_files = len(files)
    data = []
    for f in tqdm(files, total=num_files):
        img = Image.open(os.path.join(dir_path,f))
        img_array = np.array(img)
        data.append(img_array)
    return data

p = create_augmentor_pipeline(".")

# Generate  10000 images of (64 x 64) according to the pipeline and put them in `data/converted/output` folder
num_samples = 1000
p.sample(num_samples)
