from types import GeneratorType
import qrcode
from PIL import Image

def generateQR(data, filename="QRoriginal2.png"):
    qr=qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img=qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    return filename

if __name__=="__main__":
    generateQR("https://www.goole.com/")

