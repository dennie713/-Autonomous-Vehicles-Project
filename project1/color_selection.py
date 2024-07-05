import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Read in the image
image = mpimg.imread('test.jpg')

#Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

#Define color selection criteria
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
blue_threshold = 0
green_threshold = 0
red_threshold = 0
######

rgb_threshold = [blue_threshold, green_threshold, red_threshold]

# Do a boolean or with the "|" character to identify pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

#plt.rcParams["figure.figsize"] = (40,24)  #default 6.4,4.8 Width, height in inches

# Display the image                 
plt.imshow(color_select)
plt.show()