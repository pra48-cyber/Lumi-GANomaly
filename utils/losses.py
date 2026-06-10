import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class LumiLossEvaluator(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, gamma=10.0, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        self.perceptual_layers = nn.Sequential(*list(vgg.children())[:16])
        for param in self.perceptual_layers.parameters():
            param.requires_grad = False

    def forward(self, pred_frame, gt_frame, d_outputs_real, d_outputs_fake):
        hr_real, lr_real = d_outputs_real
        hr_fake, lr_fake = d_outputs_fake
        
        loss_adv = self.bce(hr_fake, torch.ones_like(hr_fake)) + self.bce(lr_fake, torch.ones_like(lr_fake))
        
        feat_pred = self.perceptual_layers(pred_frame)
        feat_gt = self.perceptual_layers(gt_frame)
        loss_perc = F.mse_loss(feat_pred, feat_gt)
        
        loss_l1 = self.l1(pred_frame, gt_frame)
        
        total_loss = (self.alpha * loss_adv) + (self.beta * loss_perc) + (self.gamma * loss_l1)
        return total_loss, loss_adv, loss_perc, loss_l1
