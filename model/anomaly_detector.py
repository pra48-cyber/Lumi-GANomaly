import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        return torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)

class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        features = x.view(batch_size, C, H * W).permute(0, 2, 1)
        att_weights = F.linear(F.normalize(features, p=2, dim=-1),
                               F.normalize(self.memory, p=2, dim=-1))
        att_weights = F.softmax(att_weights, dim=-1)
        
        if self.shrink_thres > 0:
            att_weights = torch.where(
                att_weights > self.shrink_thres,
                att_weights - self.shrink_thres,
                torch.zeros_like(att_weights)
            )
            att_weights = F.normalize(att_weights, p=1, dim=-1)
            
        output = F.linear(att_weights, self.memory.permute(1, 0))
        output = output.permute(0, 2, 1).view(batch_size, C, H, W)
        return {'output': output, 'att': att_weights}

class Generator(nn.Module):
    def __init__(self, base_channels=64, mem_dim=2000, shrink_thres=0.0025):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        self.cbam1 = CBAM(base_channels)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True)
        )
        self.cbam2 = CBAM(base_channels * 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True)
        )
        self.cbam3 = CBAM(base_channels * 4)
        
        self.mem_module = MemoryModule(mem_dim, base_channels * 4, shrink_thres)
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2 * 2, base_channels, 2, 2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        e1 = self.cbam1(self.enc1(x))
        e2 = self.cbam2(self.enc2(e1))
        e3 = self.cbam3(self.enc3(e2))
        
        mem_out = self.mem_module(e3)
        bottleneck = mem_out['output']
        
        d1 = self.dec1(bottleneck)
        d2 = self.dec2(torch.cat([d1, e2], 1))
        output = self.out_conv(torch.cat([d2, e1], 1))
        return output, mem_out['att']
