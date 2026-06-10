import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.gap(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(concat))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class HybridCNNTransformerModule(nn.Module):
    def __init__(self, channels, num_heads=8, dim_feedforward=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm(channels)
        self.msa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        feat_local = self.conv1(x)
        feat_flat = feat_local.view(b, c, h * w).permute(0, 2, 1)
        feat_norm1 = self.ln1(feat_flat)
        attn_out, _ = self.msa(feat_norm1, feat_norm1, feat_norm1)
        feat_unflat = attn_out.permute(0, 2, 1).view(b, c, h, w)
        f2 = self.ln2(self.conv2(feat_unflat).view(b, c, h * w).permute(0, 2, 1))
        f3 = self.mlp(f2)
        out = f3.permute(0, 2, 1).view(b, c, h, w)
        return x + out


class TriAttentionGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_feats=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_feats, 3, 1, 1), nn.BatchNorm2d(base_feats), nn.ReLU(True))
        self.sam1 = SpectralAttentionModule(base_feats)
        
        self.enc2 = nn.Sequential(nn.Conv2d(base_feats, base_feats * 2, 4, 2, 1), nn.BatchNorm2d(base_feats * 2), nn.ReLU(True))
        self.sam2 = SpectralAttentionModule(base_feats * 2)
        
        self.enc3 = nn.Sequential(nn.Conv2d(base_feats * 2, base_feats * 4, 4, 2, 1), nn.BatchNorm2d(base_feats * 4), nn.ReLU(True))
        self.sam3 = SpectralAttentionModule(base_feats * 4)
        
        self.cbam1 = CBAM(base_feats)
        self.cbam2 = CBAM(base_feats * 2)
        self.cbam3 = CBAM(base_feats * 4)
        
        self.hctm = HybridCNNTransformerModule(base_feats * 4)
        
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base_feats * 4, base_feats * 2, 4, 2, 1), nn.InstanceNorm2d(base_feats * 2), nn.LeakyReLU(0.2, True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_feats * 2 * 2, base_feats, 4, 2, 1), nn.InstanceNorm2d(base_feats), nn.LeakyReLU(0.2, True))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_feats * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.sam1(self.enc1(x))
        e2 = self.sam2(self.enc2(e1))
        e3 = self.sam3(self.enc3(e2))
        
        bottleneck = self.hctm(e3)
        
        d1 = self.dec1(bottleneck)
        d1_atten = torch.cat([d1, self.cbam2(e2)], dim=1)
        
        d2 = self.dec2(d1_atten)
        d2_atten = torch.cat([d2, self.cbam1(e1)], dim=1)
        
        return self.final_conv(d2_atten)


class DiscriminatorBranch(nn.Module):
    def __init__(self, in_channels=6, use_attention=False, attention_channels=128):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.feature_extractor = nn.Sequential(*layers)
        
        self.attention_block = None
        if use_attention:
            self.attention_block = HybridCNNTransformerModule(channels=128, num_heads=4, dim_feedforward=512)
            
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        if self.attention_block is not None:
            feats = self.attention_block(feats)
        return self.classifier(feats)


class DualBranchFusionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.hr_branch = DiscriminatorBranch(in_channels=6, use_attention=False)
        self.lr_branch = DiscriminatorBranch(in_channels=6, use_attention=True, attention_channels=128)

    def forward(self, real_or_predicted, conditional_frames):
        x_hr = torch.cat([real_or_predicted, conditional_frames], dim=1)
        x_lr = F.interpolate(x_hr, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        out_hr = self.hr_branch(x_hr)
        out_lr = self.lr_branch(x_lr)
        
        return out_hr, out_lr
