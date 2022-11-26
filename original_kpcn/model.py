import torch
import torch.nn.functional as F
from torch import nn
import cv2

class KPCN(nn.Module):

    def __init__(self, device, input_channels, hidden_channels=100, recon_kernel_size=21, conv_kernel_size=5, 
                 conv_depth=9, conv_kernel_nums=100):

        super(KPCN, self).__init__()
        self.device = device
        self.recon_kernel_size = recon_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_depth = conv_depth
        self.conv_kernel_nums = conv_kernel_nums
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = self.make_net()

    def make_net(self):

        # create first layer manually
        layers = [
            nn.Conv2d(self.input_channels, self.hidden_channels, self.conv_kernel_size),
            nn.ReLU()
        ]
        
        for _ in range(self.conv_depth - 2):
            layers += [
                nn.Conv2d(self.hidden_channels, self.hidden_channels, self.conv_kernel_size),
                nn.ReLU()
            ]

        out_channels = self.recon_kernel_size ** 2
        layers += [nn.Conv2d(self.hidden_channels, out_channels, self.conv_kernel_size)]
        
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
    
        return nn.Sequential(*layers)

    def apply_kernel(self, weights, data):

        # apply softmax to kernel weights
        weights = weights.permute((0, 2, 3, 1)).to(self.device)
        _, _, h, w = data.size()
        weights = F.softmax(weights, dim=3).view(-1, w * h, self.recon_kernel_size, self.recon_kernel_size)

        # now we have to apply kernels to every pixel
        # first pad the input
        r = self.recon_kernel_size // 2
        data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")

        # make slices
        R = []
        G = []
        B = []
        kernels = []
        for i in range(h):
            for j in range(w):
                pos = i * h + j
                ws = weights[:,pos:pos+1,:,:]
                kernels += [ws, ws, ws]
                sy, ey = i + r - r, i + r + r + 1
                sx, ex = j + r - r, j + r + r + 1
                R.append(data[:,0:1,sy:ey,sx:ex])
                G.append(data[:,1:2,sy:ey,sx:ex])
                B.append(data[:,2:3,sy:ey,sx:ex])
            
        reds = (torch.cat(R, dim=1).to(self.device) * weights).sum(2).sum(2)
        greens = (torch.cat(G, dim=1).to(self.device) * weights).sum(2).sum(2)
        blues = (torch.cat(B, dim=1).to(self.device) * weights).sum(2).sum(2)
        
        res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(self.device)
        
        return res

    def forward(self, x):
        return self.conv(x)