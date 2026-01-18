import os
import random
import qrcode
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2

BASE_DIR = "../Dataset"
ORIGINAL_QR_DIR = os.path.join(BASE_DIR, "Original/QR")
ORIGINAL_NONQR_DIR = os.path.join(BASE_DIR, "Original/NonQR")
AUG_TRAIN_QR = os.path.join(BASE_DIR, "Augmented/train/QR")
AUG_TRAIN_NONQR = os.path.join(BASE_DIR, "Augmented/train/NonQR")
AUG_VAL_QR = os.path.join(BASE_DIR, "Augmented/val/QR")
AUG_VAL_NONQR = os.path.join(BASE_DIR, "Augmented/val/NonQR")
AUG_TEST_QR = os.path.join(BASE_DIR, "Augmented/test/QR")
AUG_TEST_NONQR = os.path.join(BASE_DIR, "Augmented/test/NonQR")
PAIRS_QR = os.path.join(BASE_DIR, "Pairs/QR")
PAIRS_NONQR = os.path.join(BASE_DIR, "Pairs/NonQR")
for d in [
    ORIGINAL_QR_DIR, ORIGINAL_NONQR_DIR,
    AUG_TRAIN_QR, AUG_TRAIN_NONQR,
    AUG_VAL_QR, AUG_VAL_NONQR,
    AUG_TEST_QR, AUG_TEST_NONQR,
    PAIRS_QR, PAIRS_NONQR
]:
    os.makedirs(d, exist_ok=True)
def random_dark_color():
    r=random.randint(0,80)
    g=random.randint(0,80)
    b=random.randint(0,80)
    return f"#{r:02x}{g:02x}{b:02x}"

def random_light_color():
    r=random.randint(180,255)
    g=random.randint(180,255)
    b=random.randint(180,255)
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_qr(text, save_path, size=256):
    fg=random_dark_color()
    bg=random_light_color()
    qr=qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    img=qr.make_image(fill_color=fg, back_color=bg)
    img=img.resize((size, size))
    img.save(save_path)

def generate_noise_image(save_path, size=256):
    arr=np.random.randint(0,255,(size,size,3),dtype=np.uint8)
    Image.fromarray(arr).save(save_path)

def motion_blur(img):
    k=random.choice([5, 7, 9])
    kernel=np.zeros((k, k))
    kernel[:,k//2]=1
    kernel=kernel/k
    return cv2.filter2D(img,-1,kernel)


def random_perspective(img):
    h,w=img.shape[:2]
    pts1=np.float32([[0,0],[w,0],[0,h],[w,h]])
    shift=20
    pts2=np.float32([
        [random.randint(0,shift),random.randint(0,shift)],
        [w-random.randint(0,shift),random.randint(0,shift)],
        [random.randint(0,shift),h-random.randint(0,shift)],
        [w-random.randint(0,shift),h-random.randint(0,shift)]
    ])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,matrix,(w,h))


def random_background(img):
    bg_color=np.random.randint(200,255,(1,1,3),dtype=np.uint8)
    bg=np.ones_like(img)*bg_color
    mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<250
    bg[mask]=img[mask]
    return bg


def add_scratches(img):
    h,w = img.shape[:2]
    for _ in range(random.randint(1,4)):
        x1 = random.randint(0,w)
        x2 = random.randint(0,w)
        y1 = random.randint(0,h)
        y2 = random.randint(0,h)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
    return img


def augment_image(img):
    img=np.array(img)
    if random.random()<0.4:
        img=cv2.GaussianBlur(img, (3,3), 0)

    if random.random()<0.3:
        img=motion_blur(img)

    if random.random()<0.3:
        img=random_perspective(img)

    if random.random()<0.3:
        img=add_scratches(img)

    if random.random()<0.4:
        img=random_background(img)

    if random.random()<0.4:
        img=cv2.convertScaleAbs(img,alpha=random.uniform(0.6,1.4))

    if random.random()<0.4:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    if random.random()<0.4:
        noise=np.random.normal(0,12,img.shape).astype(np.uint8)
        img=cv2.add(img,noise)

    if random.random() < 0.4:
        _,enc=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(10,60)])
        img=cv2.imdecode(enc,cv2.IMREAD_COLOR)
    return Image.fromarray(img)


def generate_dataset(num_samples=300):
    for i in range(num_samples):
        qr_path=os.path.join(ORIGINAL_QR_DIR,f"qr_{i}.png")
        non_path=os.path.join(ORIGINAL_NONQR_DIR,f"non_{i}.png")
        generate_qr(f"DATA-{i}",qr_path)
        generate_noise_image(non_path)
        Image.open(qr_path).save(os.path.join(PAIRS_QR,f"qr_{i}.png"))
        Image.open(non_path).save(os.path.join(PAIRS_NONQR,f"non_{i}.png"))
        img_qr=Image.open(qr_path)
        img_non=Image.open(non_path)
        for qr_dir, non_dir in [
            (AUG_TRAIN_QR,AUG_TRAIN_NONQR),
            (AUG_VAL_QR,AUG_VAL_NONQR),
            (AUG_TEST_QR,AUG_TEST_NONQR)
        ]:
            augment_image(img_qr).save(os.path.join(qr_dir,f"qr_{i}.png"))
            augment_image(img_non).save(os.path.join(non_dir, f"non_{i}.png"))

    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_dataset(10000)
