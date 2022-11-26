import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse

from dataset import KPCNDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from ttools.modules.image_operators import crop_like
from tqdm import tqdm

from util import send_to_device, show_data_sbs
from original_kpcn.preprocess import preprocess_input, eps
from original_kpcn.sampling import importanceSampling, patch_size
from models.model import KPCN
from loss import *

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def crop(data, pos, patch_size):

    half_patch = patch_size // 2
    sx, sy = half_patch, half_patch
    px, py = pos
    return {key: val[(py-sy):(py+sy+1),(px-sx):(px+sx+1),:] 
            for key, val in data.items()}

def get_cropped_patches(sample_file, gt_file):

    data = preprocess_input(sample_file, gt_file)
    patches = importanceSampling(data)
    cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)   
    return cropped

def train(dataset, args):

    learning_rate = 1e-5
    epochs = 1000
    batch_size = 5
    print_interval = 100
    permutation = [0, 3, 1, 2]
    num_features = 16
    
    start_epoch = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    diffuse_net = KPCN(device, input_channels=num_features).to(device)
    specular_net = KPCN(device, input_channels=num_features).to(device)
    criterion = nn.L1Loss()

    if args.checkpoint >= 0:
        diffuse_checkpoint = torch.load(f'checkpoints/diffuse_checkpoint_{args.checkpoint}.pth.tar')
        specular_checkpoint = torch.load(f'checkpoints/specular_checkpoint_{args.checkpoint}.pth.tar')
        diffuse_net.load_state_dict(diffuse_checkpoint['model'])
        specular_net.load_state_dict(specular_checkpoint['model'])
        start_epoch = args.checkpoint + 1

    print('# parameters:', sum(param.numel() for param in diffuse_net.parameters()))

    diffuse_optimizer = Adam(diffuse_net.parameters(), lr=learning_rate)
    specular_optimizer = Adam(specular_net.parameters(), lr=learning_rate)

    diffuse_net.train()
    specular_net.train()

    loss_list = []

    for epoch in range(start_epoch, epochs + 1):

        total_diffuse_loss = 0
        total_specular_loss = 0
        total_loss = 0
        
        for iter, batch in enumerate(dataloader):

            # Train Diffuse Net
            diffuse_data = batch['X_diffuse'].permute(permutation).to(device)
            diffuse_data_gt = batch['Reference'][:, :, :, :3].permute(permutation).to(device)

            diffuse_optimizer.zero_grad()

            diffuse_output = diffuse_net(diffuse_data)
            diffuse_output = diffuse_net.apply_kernel(diffuse_output, crop_like(diffuse_data, diffuse_output))
            diffuse_gt = crop_like(diffuse_data_gt, diffuse_output)

            diffuse_loss = criterion(diffuse_output, diffuse_gt)

            diffuse_loss.backward()
            diffuse_optimizer.step()

            # Train Specular Net
            specular_data = batch['X_specular'].permute(permutation).to(device)
            specular_data_gt = batch['Reference'][:, :, :, 3:6].permute(permutation).to(device)

            specular_optimizer.zero_grad()

            specular_output = specular_net(specular_data)
            specular_output = specular_net.apply_kernel(specular_output, crop_like(specular_data, specular_output))
            specular_gt = crop_like(specular_data_gt, specular_output)

            specular_loss = criterion(specular_output, specular_gt)
            specular_loss.backward()
            specular_optimizer.step()

            # Calculate Final Output
            with torch.no_grad():
                albedo = batch['origAlbedo'].permute(permutation).to(device)
                albedo = crop_like(albedo, diffuse_output)
                output = diffuse_output * (albedo + eps) + torch.exp(specular_output) - 1.0

            gt = batch['finalGt'].permute(permutation).to(device)
            gt = crop_like(gt, output)
            loss = criterion(output, gt)

            total_diffuse_loss += diffuse_loss.item()
            total_specular_loss += specular_loss.item()
            total_loss += loss.item()

            loss_relative_l1 = relative_L1_loss(diffuse_output, diffuse_gt) + relative_L1_loss(specular_output, specular_gt)
            loss_relative_l2 = relative_L2_loss(diffuse_output, diffuse_gt) + relative_L2_loss(specular_output, specular_gt)
            loss_l2 = L2_loss(diffuse_output, diffuse_gt) + L2_loss(specular_output, specular_gt)
            loss_SSIM = SSIM(diffuse_output.detach().cpu().numpy(), diffuse_gt.detach().cpu().numpy()) + \
                        SSIM(specular_output.detach().cpu().numpy(), specular_gt.detach().cpu().numpy())

            loss_list.append({"l1": loss.item(), 
                              "l2": loss_l2.item(), 
                              "l1_relative": loss_relative_l1.item(),
                              "l2_relative": loss_relative_l2.item(),
                              "SSIM": loss_SSIM.item()})

            if (iter + 1) % print_interval == 0:
                
                data1 = np.clip(diffuse_output[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** 0.45454545
                data2 = np.clip(diffuse_gt[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1) ** 0.45454545
                show_data_sbs("diffuse", data1, data2)
                print(f"Epoch: [{epoch}/{epochs}] \t Iter: {iter + 1} \t Diffuse Loss: {total_diffuse_loss / (iter + 1)} \t Specular Loss: {total_specular_loss / (iter + 1)} \t Total Loss: {total_loss / (iter + 1)}")

        # Save Checkpoints
        torch.save({'model': diffuse_net.state_dict(), 'optimizer': diffuse_optimizer.state_dict(),}, f'checkpoints/diffuse_checkpoint_{epoch}.pth.tar')
        torch.save({'model': specular_net.state_dict(), 'optimizer': specular_optimizer.state_dict(),}, f'checkpoints/specular_checkpoint_{epoch}.pth.tar')
    
        with open(f"statistics/loss_{epoch}.json", "w") as f:
            f.write(json.dumps(loss_list))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=-1)
    args = parser.parse_args()

    is_debug = False
    is_loading = False

    if is_debug:

        data = preprocess_input("data/teapot_0.exr", "data/teapot_gt_0.exr", debug=is_debug)
        print(data.keys())

        patches = importanceSampling(data, debug=is_debug)
        cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)

        for i in range(5):
            data_ = np.clip(cropped[np.random.randint(0, len(cropped))]['default'], 0, 1)**0.45454545
            plt.figure(figsize = (5,5))
            imgplot = plt.imshow(data_)
            imgplot.axes.get_xaxis().set_visible(False)
            imgplot.axes.get_yaxis().set_visible(False)
            plt.savefig(f"debug_vis/patche_{i}")

        sys.exit(0)

    cropped_data = []

    if is_loading:
        data_num = len(os.listdir('train_data'))
        for i in tqdm(range(data_num), desc="Loading Data"):
            cropped_data.append(torch.load(f'train_data/sample_{i}.pt'))

    else:
        data_num = 15
        for i in tqdm(range(data_num), desc="Processing Shapenet Data"):
            for j in range(3):
                cropped_data += get_cropped_patches(f'data/dov_{i}_{j}.exr', f'data/dov_{i}_gt_{j}.exr')

        asset_dir = os.listdir(f"assets/regular")
        data_num = len(asset_dir)
        for i in tqdm(range(data_num), desc="Processing Regular Data"):
            for j in range(3):
                cropped_data += get_cropped_patches(f'data/{asset_dir[i]}_{j}.exr', f'data/{asset_dir[i]}_gt_{j}.exr')

        for i, v in enumerate(tqdm(cropped_data, desc="Saving Data")):
            torch.save(v, f'train_data/sample_{i}.pt')

    cropped_data = send_to_device(cropped_data, device)
    dataset = KPCNDataset(cropped_data)

    if is_debug:

        for i in range(5):
            sample = dataset[np.random.randint(0, len(dataset))]
            data_sample = np.clip(sample['default'], 0, 1) ** 0.45454545
            data_ref = np.clip(sample['finalGt'], 0, 1) ** 0.45454545
            show_data_sbs(i, data_sample, data_ref, figsize=(4, 2))

    train(dataset, args)