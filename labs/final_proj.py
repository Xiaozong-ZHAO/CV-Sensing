import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

# now load the image from dataset
img = cv2.imread('dataset/images/000001.png')
# show the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()