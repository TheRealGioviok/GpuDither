import cv2
import numpy as np
import matplotlib.pyplot as plt

name1= 'cpu.png'
name2= 'gpu.png'
img1 = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(name2, cv2.IMREAD_GRAYSCALE)

#create a binary map of the difference between the two images (0 - no difference, 255 - difference)
diff = cv2.absdiff(img1, img2)
_, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

#save the difference map
cv2.imwrite('diff' + name1.split("/")[-1].split(".")[0] + ".png", diff)