#testing:
import qrcode
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_base_qr(data="https://your-demo-link.com", save_path="base_qr.png"):

    qr = qrcode.QRCode(
        version=None, 
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('L')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    img.save(save_path)
    print(f"✅ Base QR Code generated at {save_path}")
    return save_path

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.msg_fc = nn.Linear(512, 64 * 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(33, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, img, msg):
        x = F.relu(self.conv1(img))
        m = self.msg_fc(msg).view(-1, 1, 64, 64)
        m = F.interpolate(m, size=(256, 256), mode='bilinear', align_corners=False)
        return img + self.conv2(torch.cat([x, m], dim=1)) * 0.5

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 64 * 64, 512)
    def forward(self, x):
        return torch.sigmoid(self.fc(self.conv(x)))


def run_v5_test(image_path, encoder_weights, decoder_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.load_state_dict(torch.load(encoder_weights, map_location=device))
    decoder.load_state_dict(torch.load(decoder_weights, map_location=device))
    
    
    encoder.eval()
    decoder.train() 
    for m in decoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    orig_pil = Image.open(image_path).convert('L')
    img_t = transform(orig_pil).unsqueeze(0).to(device)
    
    torch.manual_seed(42) 
    true_msg = torch.randint(0, 2, (1, 512)).float().to(device)

    with torch.no_grad():
        encoded_img = encoder(img_t, true_msg)
        digital_save = (encoded_img.squeeze().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        plt.imsave("V5_TO_PRINT.png", digital_save, cmap='gray')
        input_pil = Image.fromarray((digital_save * 255).astype(np.uint8))
        input_t = transform(input_pil).unsqueeze(0).to(device)
        decoded_raw = decoder(input_t)
        predicted_bits = (decoded_raw > 0.5).float()
        accuracy = (predicted_bits == true_msg).sum().item() / 512 * 100
        print(f"\n--- TEST RESULTS ---")
        print(f"✅ Recovery Accuracy: {accuracy:.2f}%")
        print(f"📊 Mean Bit Confidence: {torch.mean(torch.abs(decoded_raw - 0.5) * 2).item():.4f}")
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(digital_save, cmap='gray')
        plt.title("V5 Watermarked (Digital)")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        residual = np.abs(digital_save - (img_t.squeeze().cpu().numpy() * 0.5 + 0.5))
        plt.imshow(residual, cmap='hot')
        plt.title("Watermark Residual (Centers!)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        confidence_grid = decoded_raw.view(16, 32).cpu().numpy()
        plt.imshow(confidence_grid, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(label="Bit Confidence (0=Zero, 1=One)")
        plt.title("512-Bit Confidence Map")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ENC_WEIGHTS = "/content/drive/MyDrive/QR_Models_V5_Fresh/encoder.pth"
    DEC_WEIGHTS = "/content/drive/MyDrive/QR_Models_V5_Fresh/decoder.pth"
    BASE_IMAGE =  r"/content/qrtest7.jpeg"
    run_v5_test(BASE_IMAGE, ENC_WEIGHTS, DEC_WEIGHTS)