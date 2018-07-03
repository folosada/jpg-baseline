
# coding: utf-8

# In[1]:


#Trabalho final
#Grupo: Flávio Losada, Leonardo Fidler, Pâmela Vieira
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


def print_image_tela(image, title = ""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[4]:


def print_image(img, title = "", size = 8):
    (h,w) = img.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio,size))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


# In[82]:


def compression(imageName):
    image = cv2.imread(imageName, cv2.COLOR_BGR2YCrCb)
    
    #converte y cb cr
    #pega o y
    #aplica dct
    #quantização: usa matriz de luminancia (matriz img original - matriz luminancia)
    #rle
    #último: usa  atablea do livro gonzalez pra não precisar fazer o Huffman

    matriz_luminancia = [[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [79, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]]
    
    y,cr,cb = cv2.split(image)
    
    h,w = y.shape[:2]
    
    padding_h = 0
    padding_w = 0
    
    if (h % 8 > 0):
        padding_h = 8 - (h % 8)
        
    if (w % 8 > 0):
        padding_w = 8 - (w % 8)
    
    y = cv2.copyMakeBorder(y, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)
    
    output = np.float32(y)
    
    h,w = image.shape[:2]
    
    i = 0
    j = 0
    
    dct = cv2.dct(np.float32(y[0:8,0:8]))
    
    print(dct)    
    print(matriz_luminancia[0,0])
    
    count=0
    res = dct
    for i in range(len(dct)):
        for j in range(len(matriz_luminancia)):
            np.uint8(dct[i,j])/matriz_luminancia[i,j]
    
    print(y[0:8,0:8])
    print("dct")
    print(dct)
    print("quantization")
    print(res)
    
    i = 0
    j = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            output[i:i+8,j:j+8] = cv2.dct(np.float32(y[i:i+8,j:j+8]))
            
   
    print_image(output)


# In[83]:


compression("C:\\GitHub\\jpg-baseline\\dataset_files_download\\dataset\\teste-02.bmp")

