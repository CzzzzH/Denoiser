import torch

import numpy as np
import json
import os

from dataset import KPCNDataset, SBMCDataset, SBMCVideoDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

from ttools.modules.image_operators import crop_like
from models.models import SBMC, KPCN
from loss import TonemappedRelativeMSE, relative_L1_loss, relative_L2_loss, L1_loss, L2_loss, SSIM
from util import show_data_compare, parse_args

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f'Train Device: {device}')

def load_data(data_dir, mode='sbmc'):

    if mode == 'sbmc_video':
        train_list = []
        for i in range(1, 31):
            for j in range(1, 17):
                train_list.append(f'{data_dir}/train/shapenet_single_{i}_{j}_data.npz')
    else:
        train_dir = os.path.join(data_dir, 'train')
        train_list = os.listdir(train_dir)
        train_list = [os.path.join(train_dir, f) for f in train_list if f.endswith('.npz')]
        
    train_data = []
    for f in train_list:
        train_data.append(np.load(f))
        
    if mode == 'sbmc':
        dataset = SBMCDataset(train_data)
    elif mode == 'sbmc_video':
        dataset = SBMCVideoDataset(train_data)
    else:
        dataset = KPCNDataset(train_data)
        
    return dataset

def train_kpcn(args):
    
    num_features = 64
    
    learning_rate = 1e-5
    epochs = 100
    batch_size = 1
    print_interval = 10
    save_interval = 10
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2
    
    dataset = load_data(data_dir='data', mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    model = KPCN(n_in=num_features).to(device)
    
    start_epoch = 1001
    loss_list = []
    
    if args.checkpoint >= 0:
        start_epoch = args.checkpoint + 1
        checkpoint = torch.load(f'checkpoints/kpcn_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])
        with open(f"statistics/kpcn_loss_{args.checkpoint}.json", "r") as f:
            loss_list = json.load(f)

    print(f'Parameter Numbers: {sum(param.numel() for param in model.parameters())}')
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion  = TonemappedRelativeMSE()
    
    for epoch in range(start_epoch, epochs + 1):

        total_loss = 0
        
        for iter, (sample, gt) in enumerate(dataloader):

            # Train
            sample["kpcn_diffuse_in"] = sample["kpcn_diffuse_in"].to(device)
            sample["kpcn_specular_in"] = sample["kpcn_specular_in"].to(device)
            sample["kpcn_diffuse_buffer"] = sample["kpcn_diffuse_buffer"].to(device)
            sample["kpcn_specular_buffer"] = sample["kpcn_specular_buffer"].to(device)
            sample["kpcn_albedo"] = sample["kpcn_albedo"].to(device)
            gt = gt.permute(permutation).to(device)

            optimizer.zero_grad()

            predict = model(sample)
            gt = crop_like(gt, predict)

            loss = criterion(predict, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loss_l1 = L1_loss(predict, gt)
            loss_l2 = L2_loss(predict, gt)
            loss_relative_l1 = relative_L1_loss(predict, gt)
            loss_relative_l2 = relative_L2_loss(predict, gt)
            loss_SSIM = SSIM(predict.detach().cpu().numpy(), gt.detach().cpu().numpy())

            loss_list.append({"l1": loss_l1.item(), 
                              "l2": loss_l2.item(), 
                              "l1_relative": loss_relative_l1.item(),
                              "l2_relative": loss_relative_l2.item(),
                              "SSIM": loss_SSIM.item(), 
                              "tonemapped_l2_relative": loss.item()})
            
            # Print Loss
            if (iter + 1) % print_interval == 0:
                
                data1 = np.clip(predict[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** gamma_coeff
                data2 = np.clip(gt[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** gamma_coeff
                show_data_compare("kpcn", data1, data2)
                print(f"Epoch: [{epoch}/{epochs}] \t Iter: {iter + 1} \t Total Loss: {total_loss / (iter + 1)}")
            
        # Save Checkpoints
        if epoch % save_interval == 0:
            torch.save({'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),},
                        f'checkpoints/kpcn_checkpoint_{epoch}.pth.tar')
        
            with open(f"statistics/kpcn_loss_{epoch}.json", "w") as f:
                f.write(json.dumps(loss_list))

def train_sbmc(args):
    
    num_features = 67
    num_global_features = 3

    learning_rate = 1e-5
    epochs = 100
    batch_size = 1
    print_interval = 10
    save_interval = 10
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    if args.video:
        dataset = load_data(data_dir='data', mode='sbmc_video')
    else:
        dataset = load_data(data_dir='data', mode='sbmc')
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    # model = SBMC(n_features=num_features, n_global_features=num_global_features, video=args.video, splat=False).to(device)
    model = SBMC(n_features=num_features, n_global_features=num_global_features, video=args.video, splat=True).to(device)
    
    start_epoch = 10000
    loss_list = []
    
    if args.checkpoint >= 0:
        start_epoch = args.checkpoint + 1
        checkpoint = torch.load(f'checkpoints/sbmc_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])
        with open(f"statistics/sbmc_loss_{args.checkpoint}.json", "r") as f:
            loss_list = json.load(f)

    print(f'Parameter Numbers: {sum(param.numel() for param in model.parameters())}')
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion  = TonemappedRelativeMSE()

    for epoch in range(start_epoch, epochs + 1):

        total_loss = 0
        
        for iter, (sample, gt) in enumerate(dataloader):

            # Train
            sample["radiance"] = sample["radiance"].to(device)
            sample["features"] = sample["features"].to(device)
            sample["global_features"] = sample["global_features"].to(device)
            
            if args.video:
                gt = gt.squeeze(0)
            
            gt = gt.permute(permutation).to(device)

            optimizer.zero_grad()

            predict = model(sample)
            gt = crop_like(gt, predict)

            loss = criterion(predict, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loss_l1 = L1_loss(predict, gt)
            loss_l2 = L2_loss(predict, gt)
            loss_relative_l1 = relative_L1_loss(predict, gt)
            loss_relative_l2 = relative_L2_loss(predict, gt)
            loss_SSIM = SSIM(predict.detach().cpu().numpy(), gt.detach().cpu().numpy())

            loss_list.append({"l1": loss_l1.item(), 
                              "l2": loss_l2.item(), 
                              "l1_relative": loss_relative_l1.item(),
                              "l2_relative": loss_relative_l2.item(),
                              "SSIM": loss_SSIM.item(), 
                              "tonemapped_l2_relative": loss.item()})
            
            # Print Loss
            if (iter + 1) % print_interval == 0:
                
                if args.video:
                    frame = 8
                else:
                    frame = 0
                    
                data1 = np.clip(predict[frame].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** gamma_coeff
                data2 = np.clip(gt[frame].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** gamma_coeff
                show_data_compare("sbmc", data1, data2)
                print(f"Epoch: [{epoch}/{epochs}] \t Iter: {iter + 1} \t Total Loss: {total_loss / (iter + 1)}")
            
        # Save Checkpoints
        if epoch % save_interval == 0:
            torch.save({'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),},
                        f'checkpoints/sbmc_checkpoint_{epoch}.pth.tar')
        
            with open(f"statistics/sbmc_loss_{epoch}.json", "w") as f:
                f.write(json.dumps(loss_list))
    
if __name__ == "__main__":
    
    args = parse_args()
    if args.mode == 'sbmc':
        train_sbmc(args)
    else:
        train_kpcn(args)