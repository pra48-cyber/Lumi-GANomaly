import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.anomaly_detector import Generator
from utils.helpers import AnomalyTestDataset

class TrainConfig:
    DATA_PATH = './data/train_frames'
    SAVE_MODEL_DIR = './checkpoints'
    IMG_SIZE = 256
    BATCH_SIZE = 16
    LR = 2e-4
    EPOCHS = 50
    BASE_CHANNELS = 64
    MEM_DIM = 2000
    SHRINK_THRES = 0.0025
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    cfg = TrainConfig()
    os.makedirs(cfg.SAVE_MODEL_DIR, exist_ok=True)
    
    # Image Transformations
    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load Data
    print("Loading training dataset...")
    dataset = AnomalyTestDataset(cfg.DATA_PATH, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Initialize Model, Loss and Optimizer
    model = Generator(cfg.BASE_CHANNELS, cfg.MEM_DIM, cfg.SHRINK_THRES).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    
    print(f"Starting training loop on device: {cfg.DEVICE}")
    model.train()
    for epoch in range(cfg.EPOCHS):
        running_loss = 0.0
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(cfg.DEVICE)
            
            # Forward pass
            reconstructions, att_weights = model(inputs)
            loss = criterion(reconstructions, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] - Average Loss: {running_loss / len(dataloader):.4f}")
        
    # Save training state
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_MODEL_DIR, "best_anomaly_detector_standard.pth"))
    print("Training complete. Checkpoint saved.")
