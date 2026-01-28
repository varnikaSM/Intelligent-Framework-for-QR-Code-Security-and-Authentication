import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import random
import numpy as np
import wandb

DATASET_PATH = "/content/drive/MyDrive/NewDataset"
SAVE_PATH = "/content/drive/MyDrive/QR_Models_V5_Fresh/"
BATCH_SIZE = 16
MSG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QRDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = [f for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name).convert('L')
        return self.transform(image), torch.randint(0, 2, (MSG_SIZE,)).float()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.msg_fc = nn.Linear(512, 64 * 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(33, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh())
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

class PhysicalNoise(nn.Module):
    def forward(self, x):
        scale = random.uniform(0.7, 1.1)
        x = F.interpolate(x, scale_factor=scale, mode='bilinear')
        x = F.interpolate(x, size=(256, 256), mode='bilinear')
        
        # Simulates JPEG compression
        if random.random() > 0.5:
            x = torch.round((x * 0.5 + 0.5) * 255) / 255
            x = (x - 0.5) * 2

        # Simulates camera Blur
        if random.random() > 0.5:
            x = transforms.functional.gaussian_blur(x, [3, 3])

        return torch.clamp(x + torch.randn_like(x) * 0.02, -1, 1)

def get_edge_mask(img):
    # Uses Sobel filters to find QR module boundaries
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(DEVICE)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    edges = torch.sqrt(grad_x**2 + grad_y**2)
    return (edges > 0.5).float() # Mask is 1 at edges, 0 at centers


def train_v5():
    wandb.init(project="QR_Watermark_V5", name="Fresh_Training_CenterFocus")
    
    dataset = QRDataset(DATASET_PATH)
    loader = DataLoader(Subset(dataset, range(min(5000, len(dataset)))), batch_size=BATCH_SIZE, shuffle=True)

    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    noise_layer = PhysicalNoise().to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    print("Starting Fresh Training V5 (Edge-Avoidance Mode)...")

    for epoch in range(40):
        for img, msg in loader:
            img, msg = img.to(DEVICE), msg.to(DEVICE)
            encoded = encoder(img, msg)
            noised = noise_layer(encoded) if epoch > 5 else encoded #Curriculum learning approach
            decoded = decoder(noised)
            loss_m = F.binary_cross_entropy(decoded, msg)
            loss_i = F.mse_loss(encoded, img)
            
            # EDGE PENALTY
            mask = get_edge_mask(img)
            watermark_residual = torch.abs(encoded - img)
            loss_edge = torch.mean(watermark_residual * mask)
            total_loss = (loss_m * 600) + (loss_i * 0.5) + (loss_edge * 300)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        ber = 1.0 - ((decoded > 0.5).float() == msg).sum().item() / (msg.numel())
        print(f"Epoch {epoch+1} | BER: {ber:.4f} | Edge Penalty: {loss_edge.item():.4f}")
        wandb.log({"BER": ber, "total_loss": total_loss.item(), "edge_penalty": loss_edge.item()})

    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(SAVE_PATH, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(SAVE_PATH, "decoder.pth"))
    print(f"Models Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_v5()