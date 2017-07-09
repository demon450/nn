'''
Created on Jun 18, 2017

@author: demon
'''

import skimage
import skimage.io as io
import skimage.transform
import sys
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.image as mpimg
print("Required modules imported.")

# You can load either local IMAGE_FILE or remote URL
# For Round 1 of this tutorial, try a local image.
#IMAGE_LOCATION = 'images/cat.jpg'

# For Round 2 of this tutorial, try a URL image with a flower:
IMAGE_LOCATION = "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"

# For Round 3 of this tutorial, try another URL image with lots of people:
#IMAGE_LOCATION = "https://upload.wikimedia.org/wikipedia/commons/1/18/NASA_Astronaut_Group_15.jpg"

# For Round 4 of this tutorial, try a URL image with a portrait!
#IMAGE_LOCATION = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Ducreux1.jpg"

img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print(img.shape)

rows = 3
columns = 3
# test color reading
# show the original image
pyplot.figure()
pyplot.subplot(rows,columns,1)
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Original image = RGB')


# show the image in BGR - just doing RGB->BGR temporarily for display
imgBGR = img[:, :, (2, 1, 0)]
#pyplot.figure()
pyplot.subplot(rows,columns,2)
pyplot.imshow(imgBGR)
pyplot.axis('on')
pyplot.title('OpenCV, Caffe2 = BGR')
#pyplot.show()


'''

# Image came in sideways - it should be a portait image!
# How you detect this depends on the platform
# Could be a flag from the camera object
# Could be in the EXIF data
ROTATED_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/8/87/Cell_Phone_Tower_in_Ladakh_India_with_Buddhist_Prayer_Flags.jpg"
imgRotated = skimage.img_as_float(skimage.io.imread(ROTATED_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('on')
pyplot.title('Rotated image')

# Image came in flipped or mirrored - text is backwards!
# Again detection depends on the platform
# This one is intended to be read by drivers in their rear-view mirror
MIRROR_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/2/27/Mirror_image_sign_to_be_read_by_drivers_who_are_backing_up_-b.JPG"
imgMirror = skimage.img_as_float(skimage.io.imread(MIRROR_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('on')
pyplot.title('Mirror image')
pyplot.show()



# Run me to flip the image back and forth
imgMirror = np.fliplr(imgMirror)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('off')
pyplot.title('Mirror image')

# Run me to rotate the image 90 degrees
imgRotated = np.rot90(imgRotated)
imgRotated = np.rot90(imgRotated)
imgRotated = np.rot90(imgRotated)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('off')
pyplot.title('Rotated image')
pyplot.show()
'''

# Model is expecting 224 x 224, so resize/crop needed.
# Here are the steps we use to preprocess the image.
# (1) Resize the image to 256*256, and crop out the center.
input_height, input_width = 224, 224
print("Model's input shape is %dx%d") % (input_height, input_width)
#print("Original image is %dx%d") % (skimage.)
img2 = img
img256 = skimage.transform.resize(img2, (256, 256))
#pyplot.figure()
pyplot.subplot(rows,columns,4)
pyplot.imshow(img256)
pyplot.axis('on')
pyplot.title('Resized image to 256x256')
#pyplot.show()
print("New image shape:" + str(img256.shape))

print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
print("Model's input shape is %dx%d") % (input_height, input_width)
aspect = img.shape[1]/float(img.shape[0])
print("Orginal aspect ratio: " + str(aspect))
if(aspect>1):
    # landscape orientation - wide image
    res = int(aspect * input_height)
    imgScaled = skimage.transform.resize(img, (input_width, res))
if(aspect<1):
    # portrait orientation - tall image
    res = int(input_width/aspect)
    imgScaled = skimage.transform.resize(img, (res, input_height))
if(aspect == 1):
    imgScaled = skimage.transform.resize(img, (input_width, input_height))
#pyplot.figure()
pyplot.subplot(rows,columns,5)
pyplot.imshow(imgScaled)
pyplot.axis('on')
pyplot.title('Rescaled image')
#pyplot.show()
print("New image shape:" + str(imgScaled.shape) + " in HWC")


# Compare the images and cropping strategies
# Try a center crop on the original for giggles
print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
# yes, the function above should match resize and take a tuple...


# Original image
imgCenter = crop_center(img,224,224)
pyplot.subplot(rows,columns,7)
pyplot.imshow(imgCenter)
pyplot.axis('on')
pyplot.title('Original')

# Now let's see what this does on the distorted image
img256Center = crop_center(img256,224,224)
pyplot.subplot(rows,columns,8)
pyplot.imshow(img256Center)
pyplot.axis('on')
pyplot.title('Squeezed')

# Scaled image
imgScaledCenter = crop_center(imgScaled,224,224)
pyplot.subplot(rows,columns,9)
pyplot.imshow(imgScaledCenter)
pyplot.axis('on')
pyplot.title('Scaled')
pyplot.show()










rows = 3
columns = 2

imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)
print "Original image shape: ", imgTiny.shape
imgTiny224 = skimage.transform.resize(imgTiny, (224, 224))
print "Upscaled image shape: ", imgTiny224.shape
# Plot original
pyplot.figure()
pyplot.subplot(rows, columns, 1)
pyplot.imshow(imgTiny)
pyplot.axis('on')
pyplot.title('128x128')
# Plot upscaled
pyplot.subplot(rows, columns, 2)
pyplot.imshow(imgTiny224)
pyplot.axis('on')
pyplot.title('224x224')


print "Image shape before HWC --> CHW conversion: ", imgTiny.shape
# swapping the axes to go from HWC to CHW
# uncomment the next line and run this block!
imgTiny = imgTiny.swapaxes(1, 2).swapaxes(0, 1)
print "Image shape after HWC --> CHW conversion: ", imgTiny.shape
imgTiny224 = skimage.transform.resize(imgTiny, (224, 224))
print "Image shape after resize: ", imgTiny224.shape
# we know this is going to go wrong, so...
try:
    # Plot original
    pyplot.subplot(rows, columns, 3)
    pyplot.imshow(imgTiny)
    pyplot.axis('on')
    pyplot.title('128x128')
except:
    print "Here come bad things! look at the image shape!"
    # hands up if you want to see the error (uncomment next line)
    #raise

#reload
imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)  
imgTinySlice = crop_center(imgTiny, 128, 56)
# Plot slice
pyplot.subplot(rows, columns, 4)
pyplot.imshow(imgTinySlice)
pyplot.axis('on')
pyplot.title('128x56')
# Upscale?
print "Slice image shape: ", imgTinySlice.shape
imgTiny224 = skimage.transform.resize(imgTinySlice, (224, 224))
print "Upscaled slice image shape: ", imgTiny224.shape
# Plot upscaled
pyplot.subplot(rows, columns, 5)
pyplot.imshow(imgTiny224)
pyplot.axis('on')
pyplot.title('224x224')
pyplot.show()









# this next line helps with being able to rerun this section
# if you want to try the outputs of the different crop strategies above
# swap out imgScaled with img (original) or img256 (squeezed)
imgCropped = crop_center(imgScaled,224,224)
print "Image shape before HWC --> CHW conversion: ", imgCropped.shape
# (1) Since Caffe expects CHW order and the current image is HWC,
#     we will need to change the order.
imgCropped = imgCropped.swapaxes(1, 2).swapaxes(0, 1)
print "Image shape after HWC --> CHW conversion: ", imgCropped.shape

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(2, 3, i+1)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))



# (2) Caffe uses a BGR order due to legacy OpenCV issues, so we
#     will change RGB to BGR.
imgCropped = imgCropped[(2, 1, 0), :, :]
print "Image shape after BGR conversion: ", imgCropped.shape
# for discussion later - not helpful at this point
# (3) We will subtract the mean image. Note that skimage loads
#     image in the [0, 1] range so we multiply the pixel values
#     first to get them into [0, 255].
#mean_file = os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mean = np.load(mean_file).mean(1).mean(1)
#img = img * 255 - mean[:, np.newaxis, np.newaxis]


for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(2, 3, i+1+3)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('BGR channel %d' % (i+1))

pyplot.show()
# (4) finally, since caffe2 expect the input to have a batch term
#     so we can feed in multiple images, we will simply prepend a
#     batch dimension of size 1. Also, we will make sure image is
#     of type np.float32.
imgCropped = imgCropped[np.newaxis, :, :, :].astype(np.float32)
print 'Final input shape is:', imgCropped.shape