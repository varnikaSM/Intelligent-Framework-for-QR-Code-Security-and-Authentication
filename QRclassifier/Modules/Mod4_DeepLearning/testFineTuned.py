import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import qrcode
import numpy as np
from PIL import Image
import qrcode
import numpy as np
from PIL import Image

# --- 1. ARCHITECTURE (Character-for-character match of v3.2) ---
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


# --- USE YOUR EXISTING ENCODER/DECODER CLASSES HERE ---

def test_production_v3_2():
    device = torch.device("cpu")
    
    # Load Models
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    
    # Load your Ver3_2 weights
    enc.load_state_dict(torch.load(r"C:\Desktop\InvisibleWM\QR_Models_Ver3_2\encoder.pth", map_location=torch.device('cpu')))
    dec.load_state_dict(torch.load(r"C:\Desktop\InvisibleWM\QR_Models_Ver3_2\decoder.pth", map_location=torch.device('cpu')))

    # --- THE FIX ---
    enc.eval()
    # We keep the Decoder in a "Hybrid" mode
    dec.train() 
    # Optional: If you want to be extra safe, freeze only the weights but let BN move
    for m in dec.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False # Force it to use current batch stats
    
    # Generate Test
    qr = qrcode.make("PROD_READY_100").convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_t = transform(qr).unsqueeze(0).to(device)
    
    torch.manual_seed(42)
    msg = torch.randint(0, 2, (1, 512)).float().to(device)

    with torch.no_grad():
        # Encode
        encoded = enc(img_t, msg)
        
        # Save the watermarked image to disk
        img_np = (encoded.cpu().squeeze().numpy() * 0.5 + 0.5)
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_np).save("FINAL_WATERMARKED_QR.png")

        # Decode from the saved file (The ultimate test)
        reloaded = Image.open("FINAL_WATERMARKED_QR.png").convert('L')
        reloaded_t = transform(reloaded).unsqueeze(0).to(device)
        
        output = dec(reloaded_t)
        final_acc = ((output > 0.5).float() == msg).sum().item() / 512 * 100

    print(f"--- FINAL PRODUCTION TEST ---")
    print(f"✅ Final Extraction Accuracy: {final_acc:.2f}%")
    print(f"💾 File saved: FINAL_WATERMARKED_QR.png")

if __name__ == "__main__":
    test_production_v3_2()
'''
# --- 2. THE DIAGNOSTIC RUNNER ---
def run_local_diagnostic():
    # Update these paths to your Windows local folder
    BASE_PATH = r"C:\Desktop\InvisibleWM\QR_Models_Ver3_2"
    ENC_PATH = os.path.join(BASE_PATH, "encoder.pth")
    DEC_PATH = os.path.join(BASE_PATH, "decoder.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    
    if not os.path.exists(ENC_PATH):
        print(f"❌ ERROR: Could not find weights at {ENC_PATH}")
        return

    enc.load_state_dict(torch.load(ENC_PATH, map_location=device))
    dec.load_state_dict(torch.load(DEC_PATH, map_location=device))
    
    # --- DIAGNOSTIC STEP 1: EVAL MODE ---
    enc.eval()
    dec.eval() 

    # Prepare QR
    qr = qrcode.make("TEST_V3_2").convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_t = transform(qr).unsqueeze(0).to(device)
    
    torch.manual_seed(42)
    msg = torch.randint(0, 2, (1, 512)).float().to(device)

    with torch.no_grad():
        encoded = enc(img_t, msg)
        
        # Test A: Clean Digital
        output_clean = dec(encoded)
        acc_clean = ((output_clean > 0.5).float() == msg).sum().item() / 512 * 100
        
        # Test B: BatchNorm diagnostic (Force .train() mode)
        dec.train() 
        output_train_mode = dec(encoded)
        acc_train_mode = ((output_train_mode > 0.5).float() == msg).sum().item() / 512 * 100

    print(f"\n--- WINDOWS DIAGNOSTIC RESULTS ---")
    print(f"📊 Accuracy (Standard Eval Mode): {acc_clean:.2f}%")
    print(f"🧪 Accuracy (Forced Train Mode):  {acc_train_mode:.2f}%")
    
    if acc_clean < 60 and acc_train_mode > 90:
        print("\n💡 DIAGNOSIS: BatchNorm Collapse. Your model weights are good, but the running statistics are corrupted.")
    elif acc_clean < 60:
        print("\n💡 DIAGNOSIS: Weight Mismatch. The weights aren't communicating with this architecture.")
    else:
        print("\n✅ SUCCESS: Model is healthy.")

if __name__ == "__main__":
    run_local_diagnostic()'''

