import cv2
import pywt
import numpy as np
from QRclassifier.Modules.utils.dwtUtil import applyDWT
from QRclassifier.Modules.Mod2_URLhash.URL_binder import IdentityBinder
from QRclassifier.Modules.Mod3_Watermarking.QRcodeGen import generateQR

def embedBitsLL(LL, bitstream, alpha=10.0):
    LLembedded = LL.copy()
    meanLL = np.mean(LL) 
    height, width = LL.shape    
    if height*width<len(bitstream):
        raise ValueError(f"LL subband too small ({height}x{width}) for {len(bitstream)} bits.")
    index = 0
    for i in range(height):
        for j in range(width):
            if index >= len(bitstream):
                return LLembedded
            if bitstream[index] == '1':
                LLembedded[i, j] = meanLL + alpha
            else:
                LLembedded[i, j] = meanLL - alpha
            index += 1
            
    return LLembedded

def reconstructImage(LLembedded, LH, HL, HH, originalShape, output="QRwatermarked2.png"):
    coefficients = (LLembedded, (LH, HL, HH))
    recImage = pywt.idwt2(coefficients, 'haar')
    recImage = np.clip(recImage, 0, 255).astype(np.uint8) # Ensure values are valid 0-255 pixels
    recImage = recImage[:originalShape[0], :originalShape[1]] #padding is removed
    cv2.imwrite(output, recImage)
    return output

def QRcodeEmbedding(url, shopName, publicKeyHex, bitstream, alpha=12.0):
    qr_payload = f"URL:{url}|NAME:{shopName}|PK:{publicKeyHex}"
    imgPath="qrcodetry.png"
    tempPath=generateQR(qr_payload, imgPath)
    LL, (LH, HL, HH), origshape, _ = applyDWT(tempPath)
    LLembedded = embedBitsLL(LL, bitstream, alpha)
    outputFilename = f"SECURE{shopName}.png"
    finalImg = reconstructImage(LLembedded, LH, HL, HH, origshape,  output=outputFilename)
    return finalImg

if __name__ == "__main__":
    binder=IdentityBinder()
    private, public=binder.generateKeyPair()
    publicHex=public.public_bytes_raw().hex()
    URL="https://www.google.com"
    NAME="varnika"
    signBits=binder.create512BitWM(URL, NAME, private)
    finalQR=QRcodeEmbedding(URL, NAME, publicHex, signBits)
    print("Embededed!")
    

