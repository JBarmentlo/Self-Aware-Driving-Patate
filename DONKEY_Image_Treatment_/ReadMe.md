treatment.py 
Program to test autoencode so only a small set of images is extracted from recording
1. select images from recording 
every 100 images from DONKEY_Image/['recording_directory']
to DONKEY_Image_Treatment/DONKEY_image_Treatment/SAMPLE_10img

1. could be modified to select all images recorded

2. basic pre-processing
Resize, convert and crop image extracted from race
from directory DONKEY_Image/SAMPLE_10img to DONKEY_image_Treatment/SAMPLE_treated

program encodDecod.py
1. Increase number of images
Create images from existing ones by basic transformations 
from 'SAMPLE_treated' to 'output'
2. autoencoding Images from 'output' to 'converted'