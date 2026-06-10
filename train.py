import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.lumi_ganomaly import TriAttentionGenerator, DualBranchFusionDiscriminator
from utils.helpers import LumiAnomalyDataset
from utils.losses import LumiLossEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lumi-GANomaly Conditional GAN Framework")
    parser.add_argument("--data_path", type=str, default="./data/train_frames")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--temporal_window", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LumiAnomalyDataset(args.data_path, transform=train_transform, temporal_window=args.temporal_window, enable_enhancement=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)

    net_G = TriAttentionGenerator().to(device)
    net_D = DualBranchFusionDiscriminator().to(device)
    loss_evaluator = LumiLossEvaluator(alpha=args.alpha, beta=args.beta, gamma=args.gamma, device=device)
    
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce_loss = nn.BCELoss()

    print(f"[INFO] Training started on device: {device}")
    for epoch in range(args.epochs):
        net_G.train()
        net_D.train()
        running_loss_G, running_loss_D = 0.0, 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
        
        for inputs, _ in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            
            optimizer_D.zero_grad(set_to_none=True)
            fake_frames = net_G(inputs)
            
            d_real_hr, d_real_lr = net_D(inputs, inputs)
            loss_D_real = bce_loss(d_real_hr, torch.ones_like(d_real_hr)) + bce_loss(d_real_lr, torch.ones_like(d_real_lr))
            
            d_fake_hr, d_fake_lr = net_D(fake_frames.detach(), inputs)
            loss_D_fake = bce_loss(d_fake_hr, torch.zeros_like(d_fake_hr)) + bce_loss(d_fake_lr, torch.zeros_like(d_fake_lr))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad(set_to_none=True)
            d_outputs_real = net_D(inputs, inputs)
            d_outputs_fake = net_D(fake_frames, inputs)
            
            loss_G, _, _, _ = loss_evaluator(fake_frames, inputs, d_outputs_real, d_outputs_fake)
            loss_G.backward()
            optimizer_G.step()
            
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            progress_bar.set_postfix(G_loss=f"{loss_G.item():.4f}", D_loss=f"{loss_D.item():.4f}")
            
        print(f"[SUMMARY] Epoch [{epoch + 1}/{args.epochs}] | Avg G: {running_loss_G/len(dataloader):.4f} | Avg D: {running_loss_D/len(dataloader):.4f}")

    torch.save(net_G.state_dict(), os.path.join(args.save_dir, "lumi_generator_checkpoint.pth"))


if __name__ == "__main__":
    main()
