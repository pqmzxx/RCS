import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img=cv.imread('D://RCSgit//ball.jpg')
b=img[:,:,0]
g=img[:,:,1]
r=img[:,:,2]
rg=cv.subtract(r,g)
plt.imshow(rg,cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()