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

DATASET = r"C:\Desktop\InvisibleWM\QRclassifier\Dataset\NewDataset"
BATCHSIZE = 16
MSGSIZE = 512
EPOCHS = 20
LEARNINGRATE = 0.001

mlflow.set_experiment("QR_Invisible_WatermarkingVersion1")

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
    device = torch.device("cpu")
    dataset = QRDataset(DATASET)
    #subsetIndices = list(range(100)) 
    #trainSubset = Subset(dataset, subsetIndices)
    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    encoder = Encoder().to(device) 
    decoder = Decoder().to(device) 
    noiseLayer = NoiseLayer().to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNINGRATE)
    criterionMsg = nn.BCELoss()
    criterionImg = nn.MSELoss()

    with mlflow.start_run(run_name="CPU_Training_100"):
        mlflow.log_param("epochs", EPOCHS)
        print("Training started, view progress at http://localhost:5000")
        for epoch in range(EPOCHS):
            epochMsgLoss = 0
            epochImgLoss = 0
            for i, (img, msg) in enumerate(loader):
                encodedImg=encoder(img, msg) 
                noisedImg=noiseLayer(encodedImg)
                decodedMsg=decoder(noisedImg)
                lossI=criterionImg(encodedImg, img)
                lossM=criterionMsg(decodedMsg, msg)
                totalLoss=lossM+(lossI*10)
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()
                epochMsgLoss += lossM.item()
                epochImgLoss += lossI.item()
            
            avgMloss = epochMsgLoss / len(loader)
            avgIloss = epochImgLoss / len(loader)

            mlflow.log_metric("AverageMessageLoss", avgMloss, step=epoch)
            mlflow.log_metric("AverageImageLoss", avgIloss, step=epoch)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Msg Loss: {avgMloss:.4f} | Img Loss: {avgIloss:.4f}")
        
        mlflow.pytorch.log_model(encoder, "encoderModel")
        mlflow.pytorch.log_model(decoder, "decoderModel")
        print("Training Complete. Models Logged")

if __name__ == "__main__":
    train()