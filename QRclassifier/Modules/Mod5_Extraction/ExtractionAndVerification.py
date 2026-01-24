import cv2
import pywt
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from QRclassifier.Modules.utils.dwtUtil import applyDWT
from QRclassifier.Modules.Mod2_URLhash.URL_binder import IdentityBinder


def extractBitsLL(imagePath, bitLength=512):
    LL, (LH, HL, HH), _, _ = applyDWT(imagePath)
    global_mean = np.mean(LL)
    height, width = LL.shape
    extractedBits = ""
    index = 0
    for i in range(height):
        for j in range(width):
            if index < bitLength:
                # Decide based on global mean
                if LL[i, j] > global_mean:
                    extractedBits += '1'
                else:
                    extractedBits += '0'
                index += 1
            else: break
        if index >= bitLength: break
    return extractedBits

def verifyMerchantID(URL, shopName, publicKeyHex, extractedBits):
    try:
        binder = IdentityBinder() 
        publicBytes = bytes.fromhex(publicKeyHex)
        publicKey = ed25519.Ed25519PublicKey.from_public_bytes(publicBytes)
        signatureBytes = int(extractedBits, 2).to_bytes(64, byteorder='big')
        clean_url = binder.normalizeUrl(URL) 
        clean_name = shopName.strip()
        payload = f"{clean_url}|{clean_name}".encode('utf-8')
        publicKey.verify(signatureBytes, payload)
        return True
    except Exception as e:
        print(f"DEBUG - Crypto Error: {repr(e)}")
        return False

def scanQRdata(imagePath):
    image=cv2.imread(imagePath)
    detector=cv2.QRCodeDetector()
    data, box, straightQRCode=detector.detectAndDecode(image)
    if not data:
        raise ValueError("Could not find data in the QR Code")
    try:
        parts=data.split("|")
        url=parts[0].replace("URL:", "")
        name = parts[1].replace("NAME:", "")
        pk_hex = parts[2].replace("PK:", "")
        return url, name, pk_hex
    except IndexError:
        raise ValueError("QR data format is incorrect.")

if __name__=="__main__":
    testImage = "image.png"
    print(f"TestingQR: {testImage}")
    try:
        scannedURL, scannedName, scannedPublicKey=scanQRdata(testImage)
        print(f"Visible Data: URL: {scannedURL}, Name: {scannedName}")
        print(f"Public Key Found: {scannedPublicKey[:10]}...")
        print("Extracting DWT watermark...")
        hiddenBits = extractBitsLL(testImage)
        print(f"DEBUG - Extracted Bits (First 32): {hiddenBits[:32]}")
        print(f"DEBUG - Bit Length: {len(hiddenBits)}")
        print("Performing security verification...")
        is_valid = verifyMerchantID(scannedURL, scannedName, scannedPublicKey, hiddenBits)
        if is_valid:
            print("VERIFIED: This QR is authentic and secure.")
        else:
            print("FAILED: The watermark does not match the QR data!")

    except Exception as e:
        print(f"Test Error: {e}")


