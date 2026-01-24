import cv2
import pywt
import numpy as np
from QRclassifier.Modules.utils.dwtUtil import applyDWT
from QRclassifier.Modules.Mod2_URLhash.URL_binder import IdentityBinder
from QRclassifier.Modules.Mod3_Watermarking.QRcodeGen import generateQR

def embedBitsLL(LL, bitstream, alpha=40.0):
    LLembedded = LL.copy()
    global_mean = np.mean(LL) 
    height, width = LL.shape
    index = 0
    for i in range(height):
        for j in range(width):
            if index >= len(bitstream):
                return LLembedded
            if bitstream[index] == '1':
                LLembedded[i, j] = global_mean + alpha
            else:
                LLembedded[i, j] = global_mean - alpha
            index += 1
    return LLembedded

def reconstructImage(LLembedded, LH, HL, HH, originalShape, output="QRwatermarked2.png"):
    coefficients = (LLembedded, (LH, HL, HH))
    recImage = pywt.idwt2(coefficients, 'haar')
    recImage = np.clip(recImage, 0, 255).astype(np.uint8) 
    recImage = recImage[:originalShape[0], :originalShape[1]] 
    cv2.imwrite(output, recImage)
    return output

def QRcodeEmbedding(url, shopName, publicKeyHex, bitstream, alpha=40.0):
    qr_payload = f"URL:{url}|NAME:{shopName}|PK:{publicKeyHex}"
    imgPath="qrcodetry.png"
    tempPath=generateQR(qr_payload, imgPath)
    LL, (LH, HL, HH), origshape, _ = applyDWT(tempPath)
    LLembedded = embedBitsLL(LL, bitstream, alpha)
    outputFilename = f"SECURE{shopName}.png"
    finalImg = reconstructImage(LLembedded, LH, HL, HH, origshape,  output=outputFilename)
    return finalImg, qr_payload

if __name__ == "__main__":
    binder=IdentityBinder()
    private, public=binder.generateKeyPair()
    publicHex=public.public_bytes_raw().hex()
    print(f"pk: {public.public_bytes_raw()}")
    URL="https://www.google.com"
    NAME="varnika"
    signBits, _=binder.create512BitWM(URL, NAME, private)
    print(f"DEBUG - Original Bits (First 32): {signBits[:32]}")
    finalQR, payload=QRcodeEmbedding(URL, NAME, publicHex, signBits)
    print(payload)

    print("Embededed!")
    

