
# coding: utf-8

# In[1]:


#Trabalho final
#Grupo: Flávio Losada, Leonardo Fidler, Pâmela Vieira
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


def print_image(image, title = ""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[5]:


def compression(imageName):
    image = cv2.imread(imageName, cv2.COLOR_BGR2RGB)

    row = 0
    col = 0
    
    h,w = image.shape[:2]
    
    padding_h = 0
    padding_w = 0
    
    if (h % 8 > 0):
        padding_h = 8 - (h % 8)
        
    if (w % 8 > 0):
        padding_w = 8 - (w % 8)
    
    image = cv2.copyMakeBorder(image, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)
    
    teste = np.float32(image)
    output = np.float32(image)
    
    h,w = image.shape[:2]
    
    i = 0
    j = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            output[i:i+8,j:j+8, 0] = cv2.dct(np.float32(image[i:i+8,j:j+8, 0]))
            output[i:i+8,j:j+8, 1] = cv2.dct(np.float32(image[i:i+8,j:j+8, 1]))
            output[i:i+8,j:j+8, 2] = cv2.dct(np.float32(image[i:i+8,j:j+8, 2]))
            
    Z = output.reshape((-1,3))
    print(Z.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    quantizado = res.reshape((teste.shape))
    

    # printa imagem
    temp = quantizado
    (h,w) = temp.shape[:2]
    aspect_ratio = w/h
    size = 8
    plt.figure(figsize = (size*aspect_ratio,size))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
    plt.show()


# In[6]:


compression("C:\\GitHub\\jpg-baseline\\dataset_files_download\\dataset\\teste-02.bmp")


# In[ ]:


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

