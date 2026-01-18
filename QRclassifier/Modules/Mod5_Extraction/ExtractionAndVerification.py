import cv2
import pywt
import numpy as np
from QRclassifier.Modules.Mod2_URLhash.URL_binder import generateURLid
from QRclassifier.Modules.utils.dwtUtil import applyDWT

def extractBitsLL(LL, numBits=64):
    extractedBits = []
    meanLL = np.mean(LL)
    height, width = LL.shape
    index = 0
    
    for i in range(height):
        for j in range(width):
            if index >= numBits:
                return ''.join(extractedBits)
            if LL[i, j] > meanLL:
                extractedBits.append('1')
            else:
                extractedBits.append('0')
            index += 1
            
    return ''.join(extractedBits)

def verification(imagePath, expectedURL, numBits=64):
    LL, _, _, _, _ = applyDWT(imagePath)
    extracted = extractBitsLL(LL, numBits)
    expected = generateURLid(expectedURL)[:numBits]
    matches = sum(1 for a, b in zip(extracted, expected) if a == b)
    accuracy = (matches / numBits) * 100
    is_genuine = accuracy >= 90.0    
    return is_genuine, accuracy, extracted, expected

if __name__ == '__main__':
    image = 'QRwatermarked1.png'
    url = "https://www.google.com/"    
    is_valid, score, extracted, expected = verification(image, url)    
    print(f"\n--- VERIFICATION REPORT ---")
    print(f"Extracted: {extracted}")
    print(f"Expected:  {expected}")
    print(f"Match Score: {score:.2f}%")    
    if is_valid:
        print("QR is Genuine")
    else:
        print("QR is Tampered or Modified")