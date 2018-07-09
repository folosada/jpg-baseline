
# coding: utf-8

# In[1]:


#Trabalho final
#Grupo: Flávio Losada, Leonardo Fidler, Pâmela Vieira
import cv2 as cv2
import numpy as np
import heapq
import os
from matplotlib import pyplot as plt
from PIL import Image
import rawpy as rp
from zigzag import *


# In[2]:

MATRIX_LUMINANCE =  [[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [79, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]]
MATRIX_CHROMINANCE = [[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]]   

def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream                                                

class HeapNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq

    def __cmp__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, HeapNode)):
            return -1
        return self.freq > other.freq


# In[3]:


class HuffmanCoding:
    def __init__(self, arrayOrigin):
        self.arrayOrigin = arrayOrigin
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    def make_frequency_dict(self, array):
        frequency = {}
        for aux in array:
            if not aux in frequency:
                frequency[aux] = 0
            frequency[aux] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)


    def make_codes_helper(self, root, current_code):
        if(root == None):
            return

        if(root.value != None):
            self.codes[root.value] = current_code
            self.reverse_mapping[current_code] = root.value
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")


    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)


    def get_encoded(self, array):
        encoded = ""
        for aux in array:
            encoded += self.codes[aux]
        return encoded


    def pad_encoded(self, encoded):
        extra_padding = 8 - len(encoded) % 8
        for i in range(extra_padding):
            encoded += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded = padded_info + encoded
        return encoded


    def get_byte_array(self, padded_encoded):
        if(len(padded_encoded) % 8 != 0):
            print("Encoded not padded properly")
            exit(0)

        b = []
        for i in range(0, len(padded_encoded), 8):
            byte = padded_encoded[i:i+8]
            b.append(byte)
        return b


    def compress(self, array):
            
        frequency = self.make_frequency_dict(array)
        
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded = self.get_encoded(array)
        padded_encoded = self.pad_encoded(encoded)

        b = self.get_byte_array(padded_encoded)
        
        #print("huffman")
        #print(b)
        
        return b


# In[4]:


def print_image_tela(image, title = ""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[5]:


def print_image(img, title = "", size = 8):
    (h,w) = img.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio,size))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def reverse(image, type):
    details = image.split()
    if (type == "L"):
        matrix = MATRIX_LUMINANCE
    else:
        matrix = MATRIX_CHROMINANCE    
    # just python-crap to get integer from tokens : h and w are height and width of image (first two items)
    h = int(''.join(filter(str.isdigit, details[0])))
    w = int(''.join(filter(str.isdigit, details[1])))

    # declare an array of zeros (It helps to reconstruct bigger array on which IDCT and all has to be applied)
    array = np.zeros(h*w).astype(int)


    # some loop var initialisation
    k = 0
    i = 2
    x = 0
    j = 0


    # This loop gives us reconstructed array of size of image

    while k < array.shape[0]:
    # Oh! image has ended
        if(details[i] == ';'):
            break
    # This is imp! note that to get negative numbers in array check for - sign in string
        if "-" not in details[i]:
            array[k] = int(''.join(filter(str.isdigit, details[i])))        
        else:
            array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

        if(i+3 < len(details)):
            j = int(''.join(filter(str.isdigit, details[i+3])))

        if j == 0:
            k = k + 1
        else:                
            k = k + j + 1        

        i = i + 2

    array = np.reshape(array,(h,w))

    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    j = 0
    k = 0

    # initialisation of compressed image
    padded_img = np.zeros((h,w))

    while i < h:
        j = 0
        while j < w:        
            temp_stream = array[i:i+8,j:j+8]                
            block = inverse_zigzag(temp_stream.flatten(), int(8),int(8))            
            de_quantized = np.multiply(block,matrix)                
            padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)        
            j = j + 8        
        i = i + 8

    # clamping to  8-bit max-min values
    padded_img[padded_img > 255] = 255
    padded_img[padded_img < 0] = 0

    # compressed image is written into compressed_image.mp file
    return padded_img


# In[47]:
def compressionImage(imageName, type):
    if (type == "RAW"): 
        image = rp.imread(imageName).postprocess()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)       
    else:
        image = cv2.imread(imageName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    
    y, u, v = cv2.split(image)
    #print(y)
    #print(u)
    #print(v)
    #Adiciona uma borda na imagem caso ela não seja divisível por 8
    h,w = y.shape[:2]    
    
    padding_h = 0
    padding_w = 0
    
    if (h % 8 > 0):
        padding_h = 8 - (h % 8)
        
    if (w % 8 > 0):
        padding_w = 8 - (w % 8)
    
    y = cv2.copyMakeBorder(y, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)    
    u = cv2.copyMakeBorder(u, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)
    v = cv2.copyMakeBorder(v, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)
    
    y = compression(y, "L")
    u = compression(u, "C")
    v = compression(v, "C")    

    #newImage = cv2.merge((y, u, v)) 
    newImage = np.ndarray((y.shape[0], y.shape[1], 3), np.uint8)     
    newImage[:,:,0] = np.reshape(y, (y.shape[0], y.shape[1]))
    newImage[:,:,1] = np.reshape(u, (u.shape[0], u.shape[1]))
    newImage[:,:,2] = np.reshape(v, (v.shape[0], v.shape[1]))    
    newImage = np.reshape(newImage, (v.shape[0], v.shape[1], 3))
    #image = np.ndarray(newImage)
    print("4")
    #print_image(newImage)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_YCR_CB2BGR)
    #print_image(newImage)
    #Salva a matriz final no formato JPG
    cv2.imwrite('./dataset_files_download/dataset/sucesso_.jpeg', newImage)
    
    print("Comprimido, viva!!!")


def compression(channel, type):
    h,w = channel.shape[:2]
    hy,wy = channel.shape[:2]
    
    i = 0
    j = 0
    matrix = []
    if (type == "L"):
        matrix = MATRIX_LUMINANCE
    else:
        matrix = MATRIX_CHROMINANCE
    
    #Matriz que vai receber o resultado final
    output_matrix = channel
    output_array = []
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            output_array = []
            
            #Bloco 8x8 da imagem (subtrai zero-shift)
            image_block_8 = np.float32(channel[i:i+8,j:j+8])
            
            dct = np.float32(image_block_8)

            #parâmetros do dct: input, output, flag (faz linha por linha)
            dct = cv2.dct(image_block_8, dct, cv2.DCT_ROWS)-128
    
            #Quantização da imagem após DCT
            quantization = np.divide(np.int32(dct), matrix).astype(int)
            
            quantization = np.int32(quantization)
    
            #Algoritmo de ordenação em zig zag
            ordination = zigzag(quantization)
    
            #Codificação entrópica utilizando algoritmo de Huffman
            #huffman = HuffmanCoding(ordination)

            #result = huffman.compress(ordination)
            
            #Huffman retorna um array com números no formato binário
            #Então verifica quantidade de bits (mais o zero-shift aplicado na DCT) e aplica em uma nova matriz
            #r = 0
            #for r in range(len(result)):
            #    output_array.append(len(result[r]) + 128)
            
            #Verifica a diferença de tamanho do array de Huffman e do tamamnho do bloco 8x8
            #block_size_difference = 64 - len(result)
            
            #Completa o array com o valor 128 (zero-shift aplicado na DCT)
            #r = 0
            #if (block_size_difference > 0):
            #    for r in range(block_size_difference):
            #        output_array.append(128)
            
            #Transforma o array em uma matriz 8x8 com os valores do resultado
            #output_matrix[i:i+8,j:j+8] = np.reshape(np.asmatrix(output_array), (8,8))
            #output_matrix[i:i+8,j:j+8] = quantization
            output_matrix[i:i+8,j:j+8] = np.reshape(ordination, (8, 8)) 
    
    arranged = output_matrix.flatten()
    bitstream = get_run_length_encoding(arranged)
    bitstream = str(output_matrix.shape[0]) + " " + str(output_matrix.shape[1]) + " " + bitstream + ";"

    output_matrix = reverse(bitstream, type)
    
    return output_matrix
        
    


# In[48]:


compressionImage("./dataset_files_download/dataset/r15d1c836t.NEF", "RAW")
#compressionImage("./dataset_files_download/dataset/paisagem.bmp", "Normal")
#compression("C:\\GitHub\\jpg-baseline-bkp\\dataset_files_download\\dataset\\teste-02.bmp")

#compress("./jpg-baseline/dataset_files_download/dataset/teste-01.bmp")

