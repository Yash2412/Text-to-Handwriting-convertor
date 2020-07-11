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
lineHeight = 2*fontSize + 80


img = cv2.imread('words.jpg',0)
copy = cv2.imread('RuledPage.jpg',-1)

pageWidth = copy.shape[1]
pageHeight = copy.shape[0]
print(pageHeight,pageWidth)
pageLine = 55


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
    # temp = cv2.resize(temp,(1000,800))
    # dil = cv2.resize(dilation1,(1000,1000))
    # cv2.imshow('COmbination',temp)
    # cv2.imshow('Diale',dil)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    wor.sort(key=lambda y: y[1])
    w_ = wor[0:2] 

    w0 = wor[2:12]

    w1a = wor[12:24]
    w1b = wor[24:38]

    w2a = wor[38:50]
    w2b = wor[50:64]

    
    w_.sort(key=lambda x:x[0])
    w0.sort(key=lambda x:x[0])
    w1a.sort(key=lambda x:x[0])
    w1b.sort(key=lambda x:x[0])
    w2a.sort(key=lambda x:x[0])
    w2b.sort(key=lambda x:x[0])

    ch = ','
    for i in range(len(w_)):
        x1,y1,x2,y2 = w_[i-1][0], w_[i-1][1], w_[i-1][2], w_[i-1][3]

        words[ch] = dilation1[int(w_[i][1]):int(w_[i][3]) , int(w_[i][0]):int(w_[i][2])]
        ch = '.'

    ch  = '0'
    for i in range(len(w0)):
        x1,y1,x2,y2 = w0[i-1][0], w0[i-1][1], w0[i-1][2], w0[i-1][3]
    
        words[ch] = dilation1[int(w0[i][1]):int(w0[i][3]) , int(w0[i][0]):int(w0[i][2])]
        ch = chr(ord(ch) + 1) 

    ch = 'A'
    for i in range(len(w1a)):
        x1,y1,x2,y2 = w1a[i-1][0], w1a[i-1][1], w1a[i-1][2], w1a[i-1][3]
        if isInside(x1,y1,x2,y2,w1a[i][0],w1a[i][1]) or isInside(x1,y1,x2,y2,w1a[i][2],w1a[i][3]) :
            continue
        words[ch] = dilation1[int(w1a[i][1]):int(w1a[i][3]) , int(w1a[i][0]):int(w1a[i][2])]
        ch = chr(ord(ch) + 1) 

    for i in range(len(w1b)):
        x1,y1,x2,y2 = w1b[i-1][0], w1b[i-1][1], w1b[i-1][2], w1b[i-1][3]
        if isInside(x1,y1,x2,y2,w1b[i][0],w1b[i][1]) or isInside(x1,y1,x2,y2,w1b[i][2],w1b[i][3]) :
            continue
        words[ch] = dilation1[int(w1b[i][1]):int(w1b[i][3]) , int(w1b[i][0]):int(w1b[i][2])]
        ch = chr(ord(ch) + 1) 

    ch = 'a'
    for i in range(len(w2a)):
        x1,y1,x2,y2 = w2a[i-1][0], w2a[i-1][1], w2a[i-1][2], w2a[i-1][3]
        if isInside(x1,y1,x2,y2,w2a[i][0],w2a[i][1]) or isInside(x1,y1,x2,y2,w2a[i][2],w2a[i][3]) :
            continue
        words[ch] = dilation1[int(w2a[i][1]):int(w2a[i][3]) , int(w2a[i][0]):int(w2a[i][2])]
        ch = chr(ord(ch) + 1) 

    for i in range(len(w2b)):
        x1,y1,x2,y2 = w2b[i-1][0], w2b[i-1][1], w2b[i-1][2], w2b[i-1][3]
        if isInside(x1,y1,x2,y2,w2b[i][0],w2b[i][1]) or isInside(x1,y1,x2,y2,w2b[i][2],w2b[i][3]) :
            continue
        words[ch] = dilation1[int(w2b[i][1]):int(w2b[i][3]) , int(w2b[i][0]):int(w2b[i][2])]
        ch = chr(ord(ch) + 1) 

    
    ttt = np.ones([lineHeight,random.randint(fontSize*8, fontSize*10)], dtype = np.uint8)
    ttt[:] = 255
    words[' '] = ttt

    print(words.keys())
    
        
def plotChar():
    x = -1
    for i in words:
        x += 1
        plt.subplot(13,5,x + 1 ), plt.imshow(words[i],'gray')
        plt.title(i)
        


    plt.show()


def resizeChar():
    for i in words:
        top = random.randint(45,50)

        if i == ',' or i== '.':
            ont = np.ones([lineHeight-20,words[i].shape[1]], dtype = np.uint8)
            onb = np.ones([40,words[i].shape[1]], dtype = np.uint8)
            ont[:] = 255
            onb[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], 30))
            words[i] = np.vstack((ont,temp,onb))

        elif i >= 'A' and i<= 'Z':
            on = np.ones([top+10,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+40-top))
            words[i] = np.vstack((temp,on))

        elif i >= '0' and i<= '9':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-top))
            words[i] = np.vstack((temp,on))
            
        elif i=='b' or i=='d' or i=='f' or i=='h' or i=='k' or i=='l'  or i=='t'  :   
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
            cv2.circle(on, (v+v//2,25), 7, 0, -1)
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-top))
            words[i] = np.vstack((on,temp))
        elif i == 'i':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            v = words[i].shape[1] // 2
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-(2*top)))
            words[i] = np.vstack((on,temp,on))
            words[i] = cv2.circle(words[i], (v,25), 7, 0, -1)
        else:
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], lineHeight+50-(2*top)))
            words[i] = np.vstack((on,temp,on))


def combine(sen):
    ans = []
    w_one_char = (fontSize-10)*2+20
    h_one_char = pageLine
    one_line = 0
    line_number = 0
    output = [[] for _ in range(40)]

    for i in sen.strip().split(' '):
        if i.find('\n') != -1 :
            one_line = w_one_char * (len(i)+1)
            line_number += 1
            output[line_number] += ' '
            line_number += 1
            output[line_number] += i.replace('\n','')+' ' 

        elif one_line + w_one_char * (len(i)+1) < pageWidth - 275:
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
        tem = cv2.resize(tem,(((fontSize-10)*2+20)*len(i),pageLine))
        diff = pageWidth - 275 - tem.shape[1]
        diff1 = random.randint(0,min(20,diff))
        diff2 = diff - diff1
        on1 = np.ones([h_one_char,  diff1], dtype = np.uint8)
        on2 = np.ones([h_one_char,  diff2], dtype = np.uint8)
        on1[:] = 255
        on2[:] = 255
        tem = np.hstack((on1,tem,on2))
        ans = []

        line_space = np.ones([random.randint(13,15),  tem.shape[1]], dtype = np.uint8)
        line_space[:] = 255
        v_stack.append(tem)
        v_stack.append(line_space)

    comb = np.vstack(v_stack)
    _, comb = cv2.threshold(comb, 127, 255, cv2.THRESH_BINARY)
    comb = cv2.GaussianBlur(comb, (5,5), 0)

    # comb  = cv2.cvtColor(comb,cv2.COLOR_GRAY2BGR)
    ctem = np.ones([pageHeight,  pageWidth], dtype = np.uint8)
    ctem[:] = 255
    
    ctem[235:235+comb.shape[0],275:275+comb.shape[1]] = comb
    ctem = cv2.cvtColor(ctem,cv2.COLOR_GRAY2BGR)

    cmb = cv2.addWeighted(ctem, 0.3, copy, 0.7, 0)
    cmb  = cv2.cvtColor(cmb,cv2.COLOR_BGR2GRAY)
    _, comb = cv2.threshold(cmb, 228, 255, cv2.THRESH_BINARY)
    
    
    return comb
    # comb = cv2.GaussianBlur(comb, (5,5), 0)
    # cv2.imshow('Diale',comb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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
# plotChar()

s = sys.argv[1].replace('\r\n',' \n')
# s = s.replace('\t',' \t')

comb = combine(s)
cv2.imwrite(f.name+'\\Result.jpg', comb)

saveAsPdf()

f.cleanup()
os.remove('words.jpg')
print('I am done')

