
# coding: utf-8

# In[15]:


#Trabalho final
#Grupo: Flávio Losada, Leonardo Fidler, Pâmela Vieira
import cv2 as cv2
import numpy as np
import heapq
import os
from matplotlib import pyplot as plt
from PIL import Image


# In[16]:


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


# In[48]:


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


# In[18]:


def print_image_tela(image, title = ""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[19]:


def print_image(img, title = "", size = 8):
    (h,w) = img.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio,size))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


# In[20]:


def zigzag(array):
    
    w = len(array)
    
    d = [list(array[::-1,:].diagonal(i)[::(i+w+1)%2*-2+1])for i in range(-w,w+len(array[0]))]
    
    return sum(d,[])


# In[226]:


def compression(imageName):
    image = cv2.imread(imageName, cv2.COLOR_BGR2YCrCb)
    
    matrix_luminance =  [[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [79, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]]
    
    y, cb, cr = cv2.split(image)
    
    #print("1")
    #print(len(y))
    #print_image(y)
    
    #print("2")
    #d = cv2.merge((y, cb, cr))
    #print_image(d)
    
    h,w = y.shape[:2]
    
    print(y.shape[:2])
    
    padding_h = 0
    padding_w = 0
    
    if (h % 8 > 0):
        padding_h = 8 - (h % 8)
        
    if (w % 8 > 0):
        padding_w = 8 - (w % 8)
        
    print(padding_h)
    print(padding_w)
    
    y = cv2.copyMakeBorder(y, padding_h, 0, padding_w, 0, cv2.BORDER_REPLICATE)
    
    hy,wy = y.shape[:2]
    
    print(y.shape[:2])
    
    i = 0
    j = 0
    c=0
    
    output_array = []
    output_matrix = y

    #for i in range(0, h, 8):
        #for j in range(0, w, 8):
    while i < hy - 8:
        j = 0
        while j < wy - 8:   
            print("i: " + str(i) + " - " + str(i+8))            
            print("j: " + str(j) + " - " + str(j+8))
            
            output_array = []
            
            image_block_8 = np.float32(y[i:i+8,j:j+8])-128
            dct = np.float32(image_block_8)

            #parâmetros do dct: input, output, flag (faz linha por linha)
            dct = cv2.dct(image_block_8, dct, cv2.DCT_ROWS)
    
            quantization = np.int32(dct)/matrix_luminance
    
            quantization = np.int32(quantization)
    
            ordination = zigzag(quantization)
    
            huffman = HuffmanCoding(ordination)

            result = huffman.compress(ordination)
            
            c+=c
            
            r = 0
            for r in range(len(result)):
                output_array.append(len(result[r]) + 128)
            
            block_size_difference = 64 - len(result)
            
            #print("r: " + str(len(result)))
            #print("dif: " + str(block_size_difference))
            
            r = 0
            if (block_size_difference > 0):
                for r in range(block_size_difference):
                    output_array.append(128)
            
            output_matrix[i:i+8,j:j+8] = np.reshape(np.asmatrix(output_array), (8,8))
            
            j += 8
            
        i += 8
    
    #tem que cortar a matriz quando a imagem não é divisível por 8
    print(len(output_matrix))
    print(h)
    print(w)
    if(len(output_matrix) > len(y)):
        output_matrix = output_matrix[0:h, 0:w]
        
    print(len(output_matrix))
    
    t = cv2.merge((output_matrix, cb, cr))
    
    #newImage = cv2.cvtColor(t, cv2.COLOR_YCrCb2RGB)    
    newImage = t
    
    print("3")
    print_image(output_matrix)    
    
    print("4")
    print_image(newImage)
    
    cv2.imwrite('C:\\GitHub\\jpg-baseline-bkp\\dataset_files_download\\dataset\\sucesso.jpg', newImage)
    
    print("Comprimido, viva!!!")
    


# In[227]:


compression("C:\\GitHub\\jpg-baseline-bkp\\dataset_files_download\\dataset\\r15d1c836t.NEF")
#compress("./jpg-baseline/dataset_files_download/dataset/teste-01.bmp")

