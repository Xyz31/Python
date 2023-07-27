# Python
This repository contains all the basic concept and programs for image processing using python and its libraries.

# 1st Artithmetic
```py

import cv2
import numpy as np

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'
path2 = 'D:/6th Sem Labs/Python/5th.png'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

add = cv2.add(img1,img2)
sub = cv2.subtract(img1,img2)

cv2.imshow('Added ',add)
cv2.imshow('Subtract',sub)

cv2.imwrite('D:/6th Sem Labs/Python/2nd-addition.png',add)
cv2.imwrite('D:/6th Sem Labs/Python/2nd-subtraction.png',add)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

# 2nd Bitwise
```py

import cv2
import numpy as np

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'
path2 = 'D:/6th Sem Labs/Python/5th.png'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

bitwise_and = cv2.bitwise_and(img1,img2)
bitwise_or = cv2.bitwise_or(img1,img2)
bitwise_xor = cv2.bitwise_xor(img1,img2)
bitwise_not = cv2.bitwise_not(img1)

cv2.imshow('And',bitwise_and)
cv2.imshow('Or',bitwise_or)
cv2.imshow('Xor',bitwise_xor)
cv2.imshow('Not',bitwise_not)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

# 3rd Enhancement - Log and Histogram
```py

import cv2
import numpy as np
from matplotlib import pyplot as plt

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'

img = cv2.imread(path1)

# Log Tranformation

law1 = np.array(255*(img/255)**2.2,dtype=np.uint8)
law2 = np.array(255*(img/255)**1.2,dtype=np.uint8)

cv2.imshow('Law1',law1)
cv2.imshow('Law2',law2)


cv2.imwrite('D:/6th Sem Labs/Python/3rd-law1.png',law1)
cv2.imwrite('D:/6th Sem Labs/Python/2rd-law2.png',law2)

# 2nd part - Histogram
img = cv2.imread(path1, 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])

# Create a 2x2 grid of subplots
plt.subplot(321)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(322)
plt.imshow(equ, cmap='gray')
plt.title("Equalized")

plt.subplot(325)
plt.plot(hist, color='black')
plt.title("Histogram Original")

plt.subplot(326)
plt.plot(hist_equ, color='black')
plt.title("Histogram Equalized")

# Display the figure
plt.show()

cv2.imshow("Histogram Equalized", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()

```

# 4th Blur - Gaussian
```py

import cv2
import numpy as np
from matplotlib import pyplot as plt

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'

img = cv2.imread(path1)

blur = cv2.blur(img,(5,5))

# blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imwrite('D:/6th Sem Labs/Python/4th-Normalblurred.png',blur)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


```

# 5th Scaling and Resize
```py

import cv2
import numpy as np

path = 'C:/Users/Kashaf/Downloads/night.jpg'

img = cv2.imread(path)

resize = cv2.resize(img,(400,300))
cv2.imshow('Resiezd Image',resize)
cv2.imwrite('D:/6th Sem Labs/Python/5th.png',resize)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

# 6th Detect Edges - Canny
```py

import cv2


path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'

img = cv2.imread(path1)

edges = cv2.Canny(img, 100, 200)
cv2.imshow("Edge Detected Image", edges)
cv2.imshow("Original Image", img)

cv2.imwrite('D:/6th Sem Labs/Python/6th-Canny.png',edges)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

```

# 7th Detect Edges - Sobel and Laplacian
```py

import cv2
import numpy as np
from matplotlib import pyplot as plt

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'

img = cv2.imread(path1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# remove noise
img = cv2.GaussianBlur(gray,(3,3),0) 
# convolute with proper kernels
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray') 
plt.title('Original'), plt.xticks([]), plt.yticks([]) 

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite('D:/6th Sem Labs/Python/7th-Laplcian.png',laplacian)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray') 
plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

```

# 8th Morphological Operations - Erosion and Dilation
```py

import cv2
import numpy as np

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'


img = cv2.imread(path1, 0) # Reading the input image
kernel = np.ones((5,5), np.uint8) # Taking a matrix ofsize 5 as the kernel

img_erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imwrite('D:/6th Sem Labs/Python/8th-erosion.png',img_erosion)


#Dilation
kernel = np.ones((5,5), np.uint8)# Taking a matrix ofsize 5 as the kernel
# np.ones(shape, dtype)
# 5 x 5 is the dimension of the kernel, uint8: is

img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Input', img)
cv2.imshow('Dilation', img_dilation)
cv2.imwrite('D:/6th Sem Labs/Python/8th-Dilation.png',img_dilation)

cv2.waitKey(0)

```

# 9th Apply Mask
```py

import numpy as np
import argparse
import cv2

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'
path2 = 'D:/6th Sem Labs/Python/5th.png'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default=path2,help="path to the input image")
args = vars(ap.parse_args())
# load the original input image and display it to our screen
image = cv2.imread(path1)

cv2.imshow("Original", image)

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
cv2.imshow("Rectangular Mask", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
# now, let's make a circular mask with a radius of 100 pixels and
# apply the mask again

mask = np.zeros(image.shape[:2], dtype="uint8")

cv2.circle(mask, (145, 145), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
# show the output images
cv2.imshow("Circular Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

```

# 10th Segment - Watershed
```py

import cv2
from matplotlib import pyplot as plt
import numpy as np

path1 = 'C:/Users/Kashaf/Pictures/day2.jpg'

img = cv2.imread(path1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv2.imshow('img', img)
cv2.imwrite('D:/6th Sem Labs/Python/10th-segment.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
