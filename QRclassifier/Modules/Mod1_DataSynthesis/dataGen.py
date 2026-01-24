import qrcode
import random
import string
import os
from PIL import Image

DATASET=r"C:\Desktop\InvisibleWM\QRclassifier\Dataset\NewDataset"
NUMIMAGES = 5000
IMG_SIZE = (256, 256)
if not os.path.exists(DATASET):
    os.makedirs(DATASET)

def generate_random_data(length=30):
    chars = string.ascii_letters + string.digits + ".-/"
    return "https://secure-pay.com/verify/" + ''.join(random.choice(chars) for _ in range(length))

print(f"🚀 Generating {NUMIMAGES} unique QR codes...")

for i in range(1, NUMIMAGES + 1):
    ver = random.randint(1, 10) 
    err = random.choice([qrcode.constants.ERROR_CORRECT_L,qrcode.constants.ERROR_CORRECT_M,qrcode.constants.ERROR_CORRECT_Q,qrcode.constants.ERROR_CORRECT_H])
    qr = qrcode.QRCode(version=ver, error_correction=err, box_size=10, border=4)
    qr.add_data(generate_random_data())
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('L')
    img = img.resize(IMG_SIZE, Image.NEAREST)
    file_path = os.path.join(DATASET, f"qr_{i:04d}.png")
    img.save(file_path)
    if i % 100 == 0:
        print(f"Generated {i}/{NUMIMAGES} images...")

print(f"Done! Your dataset is ready in the '{DATASET}' folder.")