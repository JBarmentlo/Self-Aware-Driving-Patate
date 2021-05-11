from PIL import Image, ImageOps
import sys
import os
import shutil

#Image Preprocessing
myPath = "./"
for root, dirs, img in os.walk(myPath):
    for name in img:
        if '.jpg' in name:
            img = Image.open(name)
            # Size of the image in pixels (size of orginal image) 
            # (This is not mandatory) 
            width, height = img.size
            
            left = 1
            top = 40
            right = width
            bottom = 120
            
            # Cropped image of above dimension 
            # (It will not change orginal image) 
            im1 = img.crop((left, top, right, bottom)) 

            basewidth = 80
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            im1 = im1.resize((basewidth, hsize), Image.ANTIALIAS)

            gray_image = ImageOps.grayscale(im1)
            name_wo_ext, ext = name.split('.')
            gray_image.save(name_wo_ext + 'R.' + ext)
