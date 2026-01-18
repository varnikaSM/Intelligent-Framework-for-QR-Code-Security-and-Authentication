import cv2, pywt
import numpy as np

#Applying DWT
def applyDWT(imagePath):
    image=cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) #since QR codes are BW, it can be read in grayscale, 0-255 wrt intensity
    originalShape=image.shape
    image=image.astype(np.float32) 
    LL, (LH,HL,HH)=pywt.dwt2(image, "haar")
    newShape=image.shape
    return LL, (LH, HL, HH), originalShape, newShape
