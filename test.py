import torch

import numpy as np
import os
import argparse
import cv2

from dataset import KPCNDataset, SBMCDataset
from torch.utils.data import DataLoader

from ttools.modules.image_operators import crop_like
from models.models import SBMC, KPCN

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Test Device: {device}')

test_case = 'shapenet_pinhole_2_data'
visual_x = [176, 335, 305]
visual_y = [647, 269, 553]

def load_data(data_dir, mode='sbmc'):

    test_dir = os.path.join(data_dir, 'test')
    test_list = os.listdir(test_dir)
    test_list = [os.path.join(test_dir, f) for f in test_list if f.endswith('.npz')]
    test_data = []
    for f in test_list:
        if test_case in f:
            test_data.append(np.load(f))

    if mode == 'sbmc':
        dataset = SBMCDataset(test_data)
    else:
        dataset = KPCNDataset(test_data)
    return dataset

def test_kpcn(args):
    
    batch_size = 1
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    num_features = 64
    
    dataset = load_data('data', mode='kpcn')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = KPCN(n_in=num_features).to(device)

    if args.checkpoint >= 0:
        checkpoint = torch.load(f'checkpoints/kpcn_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])

    model.eval()
    
    with torch.no_grad():
        for test_index, (sample, gt) in enumerate(dataloader):

            # Test
            print(f"Test {test_index}...")
            sample["kpcn_diffuse_in"] = sample["kpcn_diffuse_in"].to(device)
            sample["kpcn_specular_in"] = sample["kpcn_specular_in"].to(device)
            sample["kpcn_diffuse_buffer"] = sample["kpcn_diffuse_buffer"].to(device)
            sample["kpcn_specular_buffer"] = sample["kpcn_specular_buffer"].to(device)
            sample["kpcn_albedo"] = sample["kpcn_albedo"].to(device)
            gt = gt.permute(permutation).to(device)
            
            predict = model(sample)
            gt = crop_like(gt, predict)

            original_img = torch.mean(sample["radiance"], dim=1)
            original_img = crop_like(original_img, predict)
            original_img = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
            predict_img = predict[0].permute(1, 2, 0).detach().cpu().numpy()
            gt_img = gt[0].permute(1, 2, 0).detach().cpu().numpy()

            original_img  = cv2.cvtColor((original_img.clip(0, np.max(original_img)) ** gamma_coeff) * 255, cv2.COLOR_BGR2RGB)
            predict_img  = cv2.cvtColor((predict_img.clip(0, np.max(predict_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)
            gt_img  = cv2.cvtColor((gt_img.clip(0, np.max(gt_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)

            if test_case == 'coffee':
                for i in range(3):
                    predict_img = cv2.circle(predict_img, (visual_x[i], visual_y[i]), 5, (0, 0, 255), -1)

            cv2.imwrite(f'test_result/test_kpcn_{test_case}_original.png', original_img)
            cv2.imwrite(f'test_result/test_kpcn_{test_case}_denoised.png', predict_img)
            cv2.imwrite(f'test_result/test_kpcn_{test_case}_reference.png', gt_img)

def test_sbmc(args):

    batch_size = 1
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    num_features = 67
    num_global_features = 3

    dataset = load_data('data', mode='sbmc')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = SBMC(n_features=num_features, n_global_features=num_global_features).to(device)

    if args.checkpoint >= 0:
        checkpoint = torch.load(f'checkpoints/sbmc_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])

    model.eval()
    
    with torch.no_grad():
        for test_index, (sample, gt) in enumerate(dataloader):

            # Test
            print(f"Test {test_index}...")
            sample["radiance"] = sample["radiance"].to(device)
            sample["features"] = sample["features"].to(device)
            sample["global_features"] = sample["global_features"].to(device)
            gt = gt.permute(permutation).to(device)
            
            predict = model(sample)
            gt = crop_like(gt, predict)

            original_img = torch.mean(sample["radiance"], dim=1)
            original_img = crop_like(original_img, predict)
            original_img = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
            predict_img = predict[0].permute(1, 2, 0).detach().cpu().numpy()
            gt_img = gt[0].permute(1, 2, 0).detach().cpu().numpy()

            original_img  = cv2.cvtColor((original_img.clip(0, np.max(original_img)) ** gamma_coeff) * 255, cv2.COLOR_BGR2RGB)
            predict_img  = cv2.cvtColor((predict_img.clip(0, np.max(predict_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)
            gt_img  = cv2.cvtColor((gt_img.clip(0, np.max(gt_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)

            if test_case == 'coffee':
                for i in range(3):
                    predict_img = cv2.circle(predict_img, (visual_x[i], visual_y[i]), 5, (0, 0, 255), -1)

            cv2.imwrite(f'test_result/test_sbmc_{test_case}_original.png', original_img)
            cv2.imwrite(f'test_result/test_sbmc_{test_case}_denoised.png', predict_img)
            cv2.imwrite(f'test_result/test_sbmc_{test_case}_reference.png', gt_img)
   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='kpcn')
    args = parser.parse_args()

    if args.mode == 'sbmc':
        test_sbmc(args)
    else:
        test_kpcn(args)