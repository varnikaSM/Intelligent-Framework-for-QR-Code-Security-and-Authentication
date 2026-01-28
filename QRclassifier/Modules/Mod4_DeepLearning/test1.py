import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import qrcode
import numpy as np
import os

# --- 1. ARCHITECTURE (Must match Training 3.1 exactly) ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.msg_fc = nn.Linear(512, 64*64)
        self.conv2 = nn.Sequential(nn.Conv2d(33, 1, 3, padding=1), nn.Tanh())

    def forward(self, img, msg):
        x = F.relu(self.conv1(img))
        m = self.msg_fc(msg).view(-1, 1, 64, 64)
        m = F.interpolate(m, size=(256, 256), mode='bilinear', align_corners=False)
        # 0.25 strength must match training
        return img + (self.conv2(torch.cat([x, m], dim=1)) * 0.25)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Flatten()
        )
        self.fc = nn.Linear(64 * 64 * 64, 512)

    def forward(self, x):
        return torch.sigmoid(self.fc(self.conv(x)))

# --- 2. PIPELINE ---
class LocalTester:
    def __init__(self, enc_path, dec_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        
        # Load weights locally
        self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(dec_path, map_location=self.device))
        
        self.encoder.eval()
        self.decoder.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def test(self, text, output_name="watermarked_local1.png"):
        # 1. Generate QR and Fixed Message
        qr = qrcode.make(text).convert('L')
        img_tensor = self.transform(qr).unsqueeze(0).to(self.device)
        
        torch.manual_seed(42) # Ensure we test with a known message
        msg = torch.randint(0, 2, (1, 512)).float().to(self.device)
        
        with torch.no_grad():
            # 2. Encode
            watermarked = self.encoder(img_tensor, msg)
            
            # Check accuracy BEFORE saving (Digital Direct)
            raw_dec = self.decoder(watermarked)
            raw_acc = ((raw_dec > 0.5).float() == msg).sum().item() / 512 * 100
            
            # 3. Save to PNG
            img_np = (watermarked.cpu().squeeze().numpy() * 0.5 + 0.5)
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_np).save(output_name)
            
            # 4. Reload from PNG
            reloaded_img = Image.open(output_name).convert('L')
            reloaded_tensor = self.transform(reloaded_img).unsqueeze(0).to(self.device)
            file_dec = self.decoder(reloaded_tensor)
            file_acc = ((file_dec > 0.5).float() == msg).sum().item() / 512 * 100

        print(f"--- Local Test Results for {output_name} ---")
        print(f"📊 Raw GPU Accuracy: {raw_acc:.2f}%")
        print(f"💾 File Extraction Accuracy: {file_acc:.2f}%")

# --- 3. RUN ---
if __name__ == "__main__":
    # UPDATE THESE TO YOUR LOCAL PATHS
    ENCODER_PATH = r"C:\Desktop\InvisibleWM\QR_Models_Ver3_1\encoder.pth"
    DECODER_PATH = r"C:\Desktop\InvisibleWM\QR_Models_Ver3_1\decoder.pth"
    
    if os.path.exists(ENCODER_PATH):
        tester = LocalTester(ENCODER_PATH, DECODER_PATH)
        tester.test("TestForThis")
    else:
        print(f"File not found: {ENCODER_PATH}")