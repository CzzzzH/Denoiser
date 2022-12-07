import torch

import numpy as np
import os
import cv2
import json

from dataset import KPCNDataset, SBMCDataset, SBMCVideoDataset
from torch.utils.data import DataLoader

from ttools.modules.image_operators import crop_like
from models.models import SBMC, KPCN
from loss import TonemappedRelativeMSE, relative_L1_loss, relative_L2_loss, L1_loss, L2_loss, SSIM
from util import parse_args

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Test Device: {device}')

test_cases = [f'shapenet_single_1_{i}' for i in range(1, 17)]
# test_cases = [f'shapenet_pinhole_{i}' for i in range(1, 11)]
# test_cases = [f'shapenet_dof_{i}' for i in range(1, 11)]
# test_cases = ['shapenet_pinhole_4']
# test_cases = ['shapenet_dof_1']
# test_cases = ['living-room']
# test_cases = ['bathroom']
# test_cases = ['veach-ajar']
# test_cases = ['teapot']
# test_cases = ['lego']
# test_cases = ['coffee']

# visual_x = [168, 327, 297]
# visual_y = [639, 261, 545]
visual_x = [176, 335, 305]
visual_y = [647, 269, 553]

def load_data(data_dir, mode='sbmc'):

    if mode == 'sbmc_video':
        test_list = []
        for i in range(1, 17):
            test_list.append(f'{data_dir}/test_video/shapenet_single_1_{i}_data.npz')
    else:
        test_dir = os.path.join(data_dir, 'test_video')
        test_list = os.listdir(test_dir)
        test_list = [os.path.join(test_dir, f'{test_case}_data.npz') for test_case in test_cases]
        
    test_data = []
    for f in test_list:
        test_data.append(np.load(f))

    if mode == 'sbmc':
        dataset = SBMCDataset(test_data)
    elif mode == 'sbmc_video':
        dataset = SBMCVideoDataset(test_data)
    else:
        dataset = KPCNDataset(test_data)
        
    return dataset

def test_kpcn(args):
    
    batch_size = 1
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    num_features = 64
    
    dataset = load_data('data', mode='kpcn')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = KPCN(n_in=num_features).to(device)
    criterion  = TonemappedRelativeMSE()
    
    if args.checkpoint >= 0:
        checkpoint = torch.load(f'checkpoints/kpcn_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])

    model.eval()
    
    avg_rmse = []
    avg_ssim = []
    
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

            loss = criterion(predict, gt).item()
            loss_SSIM = SSIM(predict.detach().cpu().numpy(), gt.detach().cpu().numpy()).item()
            loss_l2 = L2_loss(predict, gt).item()
            
            print(f'1 - SSIM: {loss_SSIM}   TonemappedRelativeMSE: {loss}    RMSE: {np.sqrt(loss_l2)}')

            original_img = torch.mean(sample["radiance"], dim=1)
            original_img = crop_like(original_img, predict)
            original_img = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
            predict_img = predict[0].permute(1, 2, 0).detach().cpu().numpy()
            gt_img = gt[0].permute(1, 2, 0).detach().cpu().numpy()

            original_img  = cv2.cvtColor((original_img.clip(0, np.max(original_img)) ** gamma_coeff) * 255, cv2.COLOR_BGR2RGB)
            predict_img  = cv2.cvtColor((predict_img.clip(0, np.max(predict_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)
            gt_img  = cv2.cvtColor((gt_img.clip(0, np.max(gt_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)

            if test_cases[0] == 'coffee':
                for i in range(3):
                    predict_img = cv2.circle(predict_img, (visual_x[i], visual_y[i]), 5, (0, 0, 255), -1)

            cv2.imwrite(f'test_result/test_kpcn_{test_cases[test_index]}_{test_index}_original.png', original_img)
            cv2.imwrite(f'test_result/test_kpcn_{test_cases[test_index]}_denoised.png', predict_img)
            cv2.imwrite(f'test_result/test_kpcn_{test_cases[test_index]}_reference.png', gt_img)

            avg_rmse.append(np.sqrt(loss_l2).item())
            avg_ssim.append(loss_SSIM)
            
    avg_rmse = np.mean(avg_rmse)
    avg_ssim = np.mean(avg_ssim)
    
    print(f"Average RMSE: {avg_rmse}")
    print(f"Average SSIM: {avg_ssim}")

def test_sbmc(args):

    batch_size = 1
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    num_features = 67
    num_global_features = 3

    if args.video:
        dataset = load_data(data_dir='data', mode='sbmc_video')
    else:
        dataset = load_data(data_dir='data', mode='sbmc')
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # model = SBMC(n_features=num_features, n_global_features=num_global_features, video=args.video, splat=False).to(device)
    model = SBMC(n_features=num_features, n_global_features=num_global_features, video=args.video, splat=True).to(device)
    criterion  = TonemappedRelativeMSE()
     
    if args.checkpoint >= 0:
        checkpoint = torch.load(f'checkpoints/sbmc_checkpoint_{args.checkpoint}.pth.tar')
        model.load_state_dict(checkpoint['model'])

    model.eval()
    
    avg_rmse = []
    avg_ssim = []
    
    with torch.no_grad():
        
        for test_index, (sample, gt) in enumerate(dataloader):

            # Test
            print(f"Test {test_index}...")
            sample["radiance"] = sample["radiance"].to(device)
            sample["features"] = sample["features"].to(device)
            sample["global_features"] = sample["global_features"].to(device)
            
            if args.video:
                gt = gt.squeeze(0)
                
            gt = gt.permute(permutation).to(device)
            
            predict = model(sample)
            gt = crop_like(gt, predict)

            if args.video:
                
                for i in range(16):
                    
                    predict_single = predict[i].unsqueeze(0)
                    gt_single = gt[i].unsqueeze(0)
                    
                    loss = criterion(predict_single, gt_single).item()
                    loss_SSIM = SSIM(predict_single.detach().cpu().numpy(), gt_single.detach().cpu().numpy()).item()
                    loss_l2 = L2_loss(predict_single, gt_single).item()
                    print(f'1 - SSIM: {loss_SSIM}   TonemappedRelativeMSE: {loss}    RMSE: {np.sqrt(loss_l2)}')
                    
                    predict_img = predict[i].permute(1, 2, 0).detach().cpu().numpy()
                    predict_img  = cv2.cvtColor((predict_img.clip(0, np.max(predict_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f'test_result/test_sbmc_shapenet_single_1_{i + 1}_denoised_video.png', predict_img)
                    avg_rmse.append(np.sqrt(loss_l2).item())
                    avg_ssim.append(loss_SSIM)
            else:
                
                loss = criterion(predict, gt).item()
                loss_SSIM = SSIM(predict.detach().cpu().numpy(), gt.detach().cpu().numpy()).item()
                loss_l2 = L2_loss(predict, gt).item()
                print(f'1 - SSIM: {loss_SSIM}   TonemappedRelativeMSE: {loss}    RMSE: {np.sqrt(loss_l2)}')

                original_img = torch.mean(sample["radiance"], dim=1)
                original_img = crop_like(original_img, predict)
                original_img = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
                predict_img = predict[0].permute(1, 2, 0).detach().cpu().numpy()
                gt_img = gt[0].permute(1, 2, 0).detach().cpu().numpy()

                original_img  = cv2.cvtColor((original_img.clip(0, np.max(original_img)) ** gamma_coeff) * 255, cv2.COLOR_BGR2RGB)
                predict_img  = cv2.cvtColor((predict_img.clip(0, np.max(predict_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)
                gt_img  = cv2.cvtColor((gt_img.clip(0, np.max(gt_img)) ** gamma_coeff)  * 255, cv2.COLOR_BGR2RGB)

                if test_cases[0] == 'coffee':
                    for i in range(len(visual_x)):
                        predict_img = cv2.circle(predict_img, (visual_x[i], visual_y[i]), 5, (0, 0, 255), -1)
                
                cv2.imwrite(f'test_result/test_sbmc_{test_cases[test_index]}_original.png', original_img)
                cv2.imwrite(f'test_result/test_sbmc_{test_cases[test_index]}_denoised.png', predict_img)
                cv2.imwrite(f'test_result/test_sbmc_{test_cases[test_index]}_reference.png', gt_img)

                avg_rmse.append(np.sqrt(loss_l2).item())
                avg_ssim.append(loss_SSIM)
    
    with open(f"statistics/sbmc_video_test_rmse.json", "w") as f:
        f.write(json.dumps(avg_rmse))
    with open(f"statistics/sbmc_video_test_ssim.json", "w") as f:
        f.write(json.dumps(avg_ssim))
    
    avg_rmse = np.mean(avg_rmse)
    avg_ssim = np.mean(avg_ssim)
    
    print(f"Average RMSE: {avg_rmse}")
    print(f"Average SSIM: {avg_ssim}")
    
if __name__ == "__main__":
    
    args = parse_args()
    if args.mode == 'sbmc':
        test_sbmc(args)
    else:
        test_kpcn(args)