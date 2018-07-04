import cv2 as cv
import numpy as np

#
#Author: Leonardo Fiedler
#
matriz_luminancia = [[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [79, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]]


zigZagMask = [
        0,  1,  8, 16,  9,  2, 3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63 ]

#Functions
def initialize(image_name):
    img = cv.imread(image_name, cv.IMREAD_COLOR)
    imgYCC = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    y, _, _ = cv.split(imgYCC)
    
    height, width = y.shape[:2]

    heightDiv = 0
    if (height % 8) > 0:
        heightDiv = 8 - (height % 8)
    
    widthDiv = 0
    if (width % 8) > 0:
        widthDiv = 8 - (width % 8)
    
    if heightDiv != 0:
        y = y[0:height - heightDiv, 0:width]
    
    if widthDiv != 0:
        y = y[0:height, 0:width - widthDiv]
    
    height, width = y.shape[:2]
    print("Height %i, Width %i" % (height, width))
    return (y, width, height)

def print_image(image, title = ""):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def compress(image_name):
    C = 1 #Compress rate - [1 Maximal - 8 Minimal]
    img, width, heigth = initialize(image_name)
    forwardDCT(C, width, heigth, img)

    return True

def forwardDCT(C, width, height, img):
    dctImage = np.zeros((height,width), np.uint8)
    
    i = 0
    mlf = np.float32(matriz_luminancia)/255.0

    while i < height - 8:
        j = 0
        while j < width - 8:
            #print(i)
            #print(j)
            rect = img[i:i+8,j:j+8]
            imf = np.float32(rect)/255.0
            dct = cv.dct(imf)
            #print(dct)
            rect = np.uint8(dct)*255.0
            div = rect / mlf
            
            add = div + 128
            add = np.uint8(add)
            result = zigzag(add)
            print(result)

            #rect = []
            #k = 0
            #res = int(C*C)
            #for k in range(0, res):
                #print("Add", add[zigZagMask[k], zigZagMask[k]])
            #    rect.append(add[zigZagMask[k], zigZagMask[k]])
                #print(rect)
            #    k += 1
            #dctImage[int(j*C/8):C, int(i*C/8):C] = rect
            #j += 8
            break
        #i += 8
        break
    print(dctImage)

    return True

# Font: https://codegolf.stackexchange.com/questions/75587/zigzagify-a-matrix
def zigzag(m):
    w=len(m)    # the height of the matrix, (at one point I thought it was the width)
    # get the anti-diagonals of the matrix. Reverse them if odd by mapping odd to -1
    d=[list(m[::-1,:].diagonal(i)[::(i+w+1)%2*-2+1])for i in range(-w,w+len(m[0]))]
            # w+len(m[0]) accounts for the width of the matrix. Works if it's too large.
    return sum(d,[]) # join the lists

# App
compress("./Trabalho Final/jpg-baseline/dataset_files_download/dataset/teste-01.bmp")
print(zigzag(np.array([[1, 2, 3], [5, 6, 4], [9, 7, 8], [1, 2, 3]])))

list = zigzag(np.array([
  [1, 2,  3,  4],
  [5, 6,  7,  8],
  [9,10, 11, 12]
]))

print(list)
