from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

words = {}
fontSize = 20
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

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    wor = []
    print('Contours', len(contours))
    # xty = cv2.drawContours(img,contours,-1,(0,200,0),2)
    # cv2.imshow('COmbination',xty)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # temp = img
    for contour in contours:
        
        x,y,w,h = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 950: #950
            continue

        # approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True) 
  
        # draws boundary of contours. 
        # cv2.drawContours(temp, [approx], 0, (0, 0, 255), 1)  
        
        # cv2.rectangle(temp, (x,y), (x+w,y+h), (0,0,0), 3)
        if x!=0 and y!=0:
            wor.append([x,y,x+w,y+h])

    # cv2.imshow('COmbination',temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    wor.sort(key=lambda y: y[3])

    w1 , w2 = [],[]
    br1 = wor[-1][1]
    for i in wor:
        if i[3] < br1:
            w1.append(i)
        else:
            w2.append(i)

    w1.sort(key=lambda x:x[0])
    w2.sort(key=lambda x:x[0])
    ch = 'a'
    for i in range(len(w1)):
        x1,y1,x2,y2 = w1[i-1][0], w1[i-1][1], w1[i-1][2], w1[i-1][3]
        if isInside(x1,y1,x2,y2,w1[i][0],w1[i][1]) or isInside(x1,y1,x2,y2,w1[i][2],w1[i][3]) :
            continue
        words[ch] = img[int(w1[i][1]):int(w1[i][3]) , int(w1[i][0]):int(w1[i][2])]
        ch = chr(ord(ch) + 1) 

    for i in range(len(w2)):
        x1,y1,x2,y2 = w2[i-1][0], w2[i-1][1], w2[i-1][2], w2[i-1][3]
        if isInside(x1,y1,x2,y2,w2[i][0],w2[i][1]) or isInside(x1,y1,x2,y2,w2[i][2],w2[i][3]) :
            continue
        words[ch] = img[int(w2[i][1]):int(w2[i][3]) , int(w2[i][0]):int(w2[i][2])]
        ch = chr(ord(ch) + 1) 

    
    ttt = np.ones([125,random.randint(70, 100)], dtype = np.uint8)
    ttt[:] = 255
    words[' '] = ttt
    
        
def plotChar():
    for i in words:
        plt.subplot(6,5,ord(i)-ord('a') + 1 ), plt.imshow(words[i],'gray')
        plt.title(i)

    plt.show()


def resizeChar():
    for i in words:
        top = random.randint(40,50)
        if i=='b' or i=='d' or i=='f' or i=='h' or i=='k' or i=='l'  or i=='t'  :   
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], 175-top))
            words[i] = np.vstack((temp,on))
        elif i=='g' or i=='p' or i=='q' or i=='y':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], 175-top))
            words[i] = np.vstack((on,temp))
        elif i == 'j':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            v = words[i].shape[1] // 2
            cv2.circle(on, (v+v//2,25), 5, 0, -1)
            temp = cv2.resize(words[i],(words[i].shape[1], 175-top))
            words[i] = np.vstack((on,temp))
        elif i == 'i':
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            v = words[i].shape[1] // 2
            temp = cv2.resize(words[i],(words[i].shape[1], 175-(2*top)))
            words[i] = np.vstack((on,temp,on))
            words[i] = cv2.circle(words[i], (v,25), 5, 0, -1)
        else:
            on = np.ones([top,words[i].shape[1]], dtype = np.uint8)
            on[:] = 255
            temp = cv2.resize(words[i],(words[i].shape[1], 175-(2*top)))
            words[i] = np.vstack((on,temp,on))


def combine(sen):
    ans = []
    w_one_char = (fontSize-10)*2+20
    h_one_char = (fontSize-10)*2+75
    one_line = 0
    line_number = 0
    output = [[] for _ in range(30)]

    for i in sen.strip().split(' '):
        if one_line + w_one_char * (len(i)+1) < pageWidth - 500:
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
        tem = cv2.resize(tem,(((fontSize-10)*2+20)*len(i),(fontSize-10)*2+75))
        diff = pageWidth - 500 - tem.shape[1]
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

    
    copy[250:250+comb.shape[0],250:250+comb.shape[1]] = comb
    return comb

# cv2.imshow('COmbination',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

getWords()
resizeChar()
# plotChar()

comb = combine(sys.argv[1].lower())
cv2.imwrite('./views/Result.jpg',copy)

print('I am done')
# Lorem Ipsum is simply dummy text of the printing and typesetting industry Lorem Ipsum has been the 
# industrys standard dummy text ever since the fifteen hundred when an unknown printer took a galley of type and scrambled it 
# to make a type specimen book It has survived not only five centuries but also the leap into electronic typesetting