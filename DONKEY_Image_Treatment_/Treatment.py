from PIL import Image, ImageOps
import sys
import os
import shutil

alReadyDone = 1

if alReadyDone == 1:
    """ 
        Compter toutes images
    """
    jpgCounter = 0
    myPath = "../DONKEY_Image/TEST3"

    for root, dirs, files in os.walk(myPath):
        for name in files:    
            if name.endswith('.jpg'):
                jpgCounter += 1
    print("jpg files number {}".format(jpgCounter))
    # len(os.walk(path).next()[2]) : nombre total de fichiers

    # Selection de 10 images
    for root, dirs, files in os.walk(myPath):
        for name in files:    
            for num_img in range(0,10000,100):
                if name.startswith(str(num_img)+"_"):
                    shutil.copyfile(r"../DONKEY_Image/TEST3/"+name, r"../DONKEY_Image_Treatment/SAMPLE_10img/"+name)   
#Image Preprocessing
myPath = "../DONKEY_Image_Treatment/SAMPLE_10img/"
for root, dirs, img in os.walk(myPath):
    for name in img:
        img = Image.open(myPath + name)
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
        # gray_image.show()
        path_treated = "../DONKEY_Image_Treatment/SAMPLE_treated/"
        name_wo_ext, ext = name.split('.')
        gray_image.save(path_treated + name_wo_ext + 'R.' + ext)
