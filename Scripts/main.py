from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import tempfile

import img2pdf 
from PIL import Image 
import os 


words = {}
fontSize = int(sys.argv[2])
lineHeight = 2*fontSize + 100
pageWidth = 2481
pageHeight = 3507

img = cv2.imread('words.jpg',0)
copy = np.ones([pageHeight, pageWidth],dtype = np.uint8)
copy[:]=255


def isInside(x1, y1, x2, y2, x, y) : 
    if (x > x1 and x < x2 and y > y1 and y < y2) : 
        return True
    else : 
        return False


def getWords():
    global words

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
  
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1) 
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    wor = []
    print('Contours', len(contours))
    # xty = cv2.drawContours(img,contours,-1,(0,200,0),2)
    # cv2.imshow('COmbination',xty)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    temp = img.copy()
    for contour in contours:
        
        x,y,w,h = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1200: #950
            continue

        # approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True) 
  
        # draws boundary of contours. 
        # cv2.drawContours(temp, [approx], 0, (0, 0, 255), 1)  
        
        # cv2.rectangle(temp, (x,y), (x+w,y+h), (255,255,255), 5)
        if x!=0 and y!=0:
            wor.append([x,y,x+w,y+h])
    
    
    
    
    rect_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) 
    dilation1 = cv2.dilate(thresh, rect_kernel1, iterations = 1)

    rect_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    dilation1 = cv2.erode(dilation1, rect_kernel1, iterations=1) 


    dilation1 = cv2.subtract(255, dilation1)

     
    # thresh =  cv2.subtract(255, thresh)
    # temp = cv2.resize(thresh,(1000,1500))
    # dil = cv2.resize(dilation1,(1000,1500))
    # cv2.imshow('COmbination',temp)
    # cv2.imshow('Diale',dil)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    wor.sort(key=lambda y: y[1])

    w1 = wor[:12]
    w2 = wor[12:]
    

    w1.sort(key=lambda x:x[0])
    w2.sort(key=lambda x:x[0])
    ch = 'a'
    for i in range(len(w1)):
        x1,y1,x2,y2 = w1[i-1][0], w1[i-1][1], w1[i-1][2], w1[i-1][3]
        if isInside(x1,y1,x2,y2,w1[i][0],w1[i][1]) or isInside(x1,y1,x2,y2,w1[i][2],w1[i][3]) :
            continue
        words[ch] = dilation1[int(w1[i][1]):int(w1[i][3]) , int(w1[i][0]):int(w1[i][2])]
        ch = chr(ord(ch) + 1) 

    for i in range(len(w2)):
        x1,y1,x2,y2 = w2[i-1][0], w2[i-1][1], w2[i-1][2], w2[i-1][3]
        if isInside(x1,y1,x2,y2,w2[i][0],w2[i][1]) or isInside(x1,y1,x2,y2,w2[i][2],w2[i][3]) :
            continue
        words[ch] = dilation1[int(w2[i][1]):int(w2[i][3]) , int(w2[i][0]):int(w2[i][2])]
        ch = chr(ord(ch) + 1) 

    
    ttt = np.ones([lineHeight,random.randint(fontSize*10, fontSize*13)], dtype = np.uint8)
    ttt[:] = 255
    words[' '] = ttt

    print(words.keys())
    
        
def plotChar():
    for i in words:
        if(i != ' '):
            plt.subplot(6,5,ord(i)-ord('a') + 1 ), plt.imshow(words[i],'gray')
            plt.title(i)
        else:
            plt.subplot(6,5,30 ), plt.imshow(words[i],'gray')
            plt.title(i)


    plt.show()


def resizeChar():
    for i in words:
        top = random.randint(40,50)
        if i=='b' or i=='d' or i=='f' or i=='h' or i=='k' or i=='l'  or i=='t'  :   
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-top))
            words[i] = np.vstack((temp,on))
        elif i=='g' or i=='p' or i=='q' or i=='y':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-top))
            words[i] = np.vstack((on,temp))
        elif i == 'j':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            v = words[i].shape[1] // 2
            # cv2.circle(on, (v+v//2,25), 7, 0, -1)
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-top))
            words[i] = np.vstack((on,temp))
        elif i == 'i':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            v = words[i].shape[1] // 2
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-(2*top)))
            words[i] = np.vstack((on,temp,on))
            # words[i] = cv2.circle(words[i], (v,25), 7, 0, -1)
        else:
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-(2*top)))
            words[i] = np.vstack((on,temp,on))


def combine(sen):
    ans = []
    w_one_char = (fontSize-10)*2+20
    h_one_char = (fontSize-10)*2+85
    one_line = 0
    line_number = 0
    output = [[] for _ in range(30)]

    for i in sen.strip().split(' '):
        if one_line + w_one_char * (len(i)+1) < pageWidth - 400:
            one_line += w_one_char * (len(i)+1)
            output[line_number] += i+' '
        else:
            one_line = w_one_char * (len(i)+1)
            line_number += 1
            output[line_number] += i+' ' 
    
    v_stack = []

    for i in output:
        if(len(i) == 0):
            continue
        for j in list(i):
            ans.append(words[j])
        tem = np.hstack(ans)
        tem = cv2.resize(tem,(((fontSize-10)*2+20)*len(i),(fontSize-10)*2+85))
        diff = pageWidth - 400 - tem.shape[1]
        diff1 = random.randint(0,min(40,diff))
        diff2 = diff - diff1
        on1 = np.ones([h_one_char,  diff1], dtype = np.uint8)
        on2 = np.ones([h_one_char,  diff2], dtype = np.uint8)
        on1[:] = 255
        on2[:] = 255
        tem = np.hstack((on1,tem,on2))
        ans = []
        v_stack.append(tem)

    comb = np.vstack(v_stack)
    _, comb = cv2.threshold(comb, 127, 255, cv2.THRESH_BINARY)
    comb = cv2.GaussianBlur(comb, (5,5), 0)

    
    copy[200:200+comb.shape[0],200:200+comb.shape[1]] = comb


f = tempfile.TemporaryDirectory(dir = os.getcwd())



def saveAsPdf():
    img_path = f.name+"\\Result.jpg"
 
    pdf_path = "./views/Result.pdf"

    image = Image.open(img_path) 

    pdf_bytes = img2pdf.convert(image.filename) 
    
    file = open(pdf_path, "wb") 
    
    file.write(pdf_bytes) 
    
    image.close() 
    
    file.close() 


# cv2.imshow('COmbination',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

getWords()
resizeChar()
plotChar()

combine(sys.argv[1].lower())
cv2.imwrite(f.name+'\\Result.jpg', copy)

saveAsPdf()

f.cleanup()
os.remove('words.jpg')
print('I am done')

