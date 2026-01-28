import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
from PIL import Image
import mlflow
import mlflow.pytorch
#from google.colab import files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "/content/drive/MyDrive/NewDataset"
BATCHSIZE = 16
MSGSIZE = 512
EPOCHS = 20
LEARNINGRATE = 0.001
mlflow.set_experiment("QR_Invisible_Watermarking_Colab")
class QRDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = [f for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imgName = os.path.join(self.root, self.images[idx])
        image = Image.open(imgName).convert('L')
        image = self.transform(image)
        message = torch.randint(0, 2, (MSGSIZE,)).float()
        return image, message

class NoiseLayer(nn.Module):
    def __init__(self):
        super(NoiseLayer, self).__init__()

    def forward(self, x):
        noise = torch.randn_like(x) * 0.05
        x = x + noise 
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) 
        shiftx, shifty = torch.randint(-2, 3, (2, ))
        x = torch.roll(x, shifts=(shiftx, shifty), dims=(2, 3))
        return torch.clamp(x, -1, 1)

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
        self.fc = nn.Linear(64 * 64 * 64, MSGSIZE)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return torch.sigmoid(x)

def train():
    # 1. Initialize W&B
    wandb.init(
        project="QR_Invisible_Watermarking",
        config={
            "learning_rate": LEARNINGRATE,
            "epochs": EPOCHS,
            "batch_size": BATCHSIZE,
            "msg_size": MSGSIZE,
            "dataset_size": 5000
        }
    )

    dataset = QRDataset(DATASET)
    # Subset to 5000 images for better generalization
    subset_indices = list(range(min(5000, len(dataset))))
    train_subset = Subset(dataset, subset_indices)
    loader = DataLoader(train_subset, batch_size=BATCHSIZE, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    noise_layer = NoiseLayer().to(device)
    
    # Optional: Log the model architecture/gradients
    wandb.watch(encoder, log_freq=100)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNINGRATE)
    criterion_msg = nn.BCELoss()
    criterion_img = nn.MSELoss()

    print(f"Training on {device} with 5,000 images...")

    for epoch in range(EPOCHS):
        epoch_msg_loss = 0
        epoch_img_loss = 0
        total_correct_bits = 0
        total_bits = 0
        
        for img, msg in loader:
            img, msg = img.to(device), msg.to(device)

            encoded_img = encoder(img, msg)
            noised_img = noise_layer(encoded_img)
            decoded_msg = decoder(noised_img)

            loss_i = criterion_img(encoded_img, img)
            loss_m = criterion_msg(decoded_msg, msg)
            
            # --- THE FIX: Increase message weight ---
            # We want the model to be 'punished' more for failing the message
            total_loss = (loss_m * 50) + (loss_i * 1) 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- BIT ERROR RATE (BER) CALCULATION ---
            # Round the sigmoid output to 0 or 1
            predicted_bits = (decoded_msg > 0.5).float()
            correct_bits = (predicted_bits == msg).float().sum().item()
            
            total_correct_bits += correct_bits
            total_bits += msg.numel()
            
            epoch_msg_loss += loss_m.item()
            epoch_img_loss += loss_i.item()

        avg_m_loss = epoch_msg_loss / len(loader)
        avg_i_loss = epoch_img_loss / len(loader)
        
        # Calculate BER (0.0 is perfect, 0.5 is random guessing)
        ber = 1.0 - (total_correct_bits / total_bits)
        
        # 2. Log Metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "message_loss": avg_m_loss,
            "image_loss": avg_i_loss,
            "bit_error_rate": ber,
            "total_loss": total_loss.item()
        })

        print(f"Epoch [{epoch+1}/{EPOCHS}] - BER: {ber:.4f} | Msg Loss: {avg_m_loss:.4f} | Img Loss: {avg_i_loss:.4f}")
    # 3. Save models to Google Drive
    save_path = "/content/drive/MyDrive/QR_Models/"
    os.makedirs(save_path, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_path, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_path, "decoder.pth"))
      
    # Also save to W&B cloud
    wandb.save(os.path.join(save_path, "*.pth"))
    wandb.finish()
    print("Training Complete. Models saved to Drive and W&B.")
if __name__ == "__main__":
    train()