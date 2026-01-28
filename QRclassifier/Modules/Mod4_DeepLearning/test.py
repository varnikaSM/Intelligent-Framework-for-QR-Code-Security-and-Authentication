import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import qrcode
import os

# --- CONFIGURATION ---
MSGSIZE = 512
# Path to where your .pth files are actually stored on your Windows machine
MODEL_DIR = r"C:\Desktop\InvisibleWM\QR_Models" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARCHITECTURE (MUST MATCH YOUR TRAINING CODE) ---

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.msg_fc = nn.Linear(MSGSIZE, 64*64)
        self.conv2 = nn.Conv2d(33, 1, 3, padding=1)

    def forward(self, img, msg):
        x = F.relu(self.conv1(img))
        m = self.msg_fc(msg).view(-1, 1, 64, 64)
        m = F.interpolate(m, size=(256, 256))
        x = torch.cat([x, m], dim=1)
        x = self.conv2(x)
        return x + img

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten()
        )
        # Matches the 64 * 64 * 64 from your training script
        self.fc = nn.Linear(64 * 64 * 64, MSGSIZE)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# --- UTILS ---

def load_models():
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    
    enc_path = os.path.join(MODEL_DIR, "encoder.pth")
    dec_path = os.path.join(MODEL_DIR, "decoder.pth")
    
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    dec.load_state_dict(torch.load(dec_path, map_location=device))
    
    enc.eval()
    dec.eval()
    return enc, dec

def test_watermark(data_text="https://google.com"):
    encoder, decoder = load_models()

    # 1. Generate QR
    qr = qrcode.make(data_text).convert('L')
    
    # 2. Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(qr).unsqueeze(0).to(device)
    
    # 3. Create a message (using same seed for verification)
    torch.manual_seed(42)
    secret_msg = torch.randint(0, 2, (1, MSGSIZE)).float().to(device)

    with torch.no_grad():
        # Encode
        watermarked_img = encoder(img_tensor, secret_msg)
        
        # Decode
        extracted_msg = decoder(watermarked_img)
        predicted_bits = (extracted_msg > 0.5).float()
        
        # Calculate Accuracy
        correct = (predicted_bits == secret_msg).sum().item()
        accuracy = (correct / MSGSIZE) * 100
        
        print(f"\n--- TEST RESULTS ---")
        print(f"Accuracy: {accuracy:.2f}%")
        if accuracy > 90:
            print("Status: Success! The architectures are synced.")
        else:
            print("Status: Still seeing low accuracy. Check if the .pth files are definitely from the latest run.")

if __name__ == "__main__":
    test_watermark()