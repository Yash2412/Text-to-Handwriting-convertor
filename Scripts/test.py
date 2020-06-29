import img2pdf 
from PIL import Image 
import numpy as np
import os 
from cv2 import cv2
import tempfile
f = tempfile.TemporaryDirectory(dir = os.getcwd())

copy = cv2.imread('Result.jpg',0)




def saveAsPdf():
    img_path = f.name+"\\Result.jpg"
 
    pdf_path = "./views/file.pdf"

    image = Image.open(img_path) 

    pdf_bytes = img2pdf.convert(image.filename) 
    
    file = open(pdf_path, "wb") 
    
    file.write(pdf_bytes) 
    
    image.close() 
    
    file.close() 


cv2.imwrite(f.name+'\\Result.jpg', copy)

saveAsPdf()

f.cleanup()