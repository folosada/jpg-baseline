
# coding: utf-8

# In[1]:


#Trabalho final
#Grupo: Flávio Losada, Leonardo Fidler, Pâmela Vieira
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# In[6]:


def print_image(img, title = "", size = 8):
    (h,w) = img.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio,size))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def setParamCompression(imageName):
    image = cv2.imread(imageName, cv2.IMREAD_COLOR)
    newImage = imageName[:len(imageName)-4] + ".jpg"
    fileImage = open(newImage, 'wb')

    height, width, channels = image.shape
    
    heightToMultple8 = 0 if (height % 8 == 0) else (8 - height % 8)
    widthToMultple8 = 0 if (width % 8 == 0) else (8 - width % 8)
    
    if (heightToMultple8 != 0):
        block = image[0:height-heightToMultple8, 0:width-widthToMultple8]
        height, width, channels = block.shape
        print(height)
        print(width)
        cv2.imshow("test", block)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(heightToMultple8)
    print(widthToMultple8)

# In[10]:

"""
#img = cv2.imread('C:/Furb/satelite1.png', cv2.IMREAD_COLOR)
img = cv2.imread('C:/Furb/satelite1.png', 0)      # 1 chan, grayscale!

#Transformada DCT - link: http://answers.opencv.org/question/9578/how-to-get-dct-of-an-image-in-python-using-opencv/
imf = np.float32(img)/255.0  # float conversion/scale
dct = cv2.dct(imf)           # the dct
#img = np.uint8(dct)*255.0    # convert back
#print_image(dct)

#Quantização - link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(dct,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#Codificação RLE


#Codificação Estatístico

"""

setParamCompression("./dataset_files_download/dataset/teste-02.bmp")