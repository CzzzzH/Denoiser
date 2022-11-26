import torch
import numpy as np
import cv2
import torch.nn.functional as F

from torch import nn
from ttools.modules.image_operators import crop_like

from util import send_to_device, to_torch_tensors, show_data, unsqueeze_all
from original_kpcn.preprocess import preprocess_input, eps
from models.model import KPCN
from loss import RMSE, SSIM

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_kernel = True
visual_x = [168, 327, 297]
visual_y = [639, 261, 545]

def save_kernel(kernel, kernel_size=21, type='diffuse'):
    
    _, _, h, w = kernel.size()
    kernel = kernel.permute((0, 2, 3, 1))
    kernel = F.softmax(kernel, dim=3).view(-1, h, w, kernel_size, kernel_size)[0].cpu().numpy()

    for i in range(3):
        k = kernel[visual_y[i], visual_x[i]]
        k = (k - k.min()) / (k.max() - k.min())
        k = cv2.resize(k, (kernel_size * 10, kernel_size * 10))
        cv2.imwrite(f"test_result/kpcn_kernel_{type}_{visual_x[i]}_{visual_y[i]}.png", k * 255)

def test(data, scene_name, diffuse_checkpoint_path, specular_checkpoint_path):

    num_features = 16
    permutation = [0, 3, 1, 2]
    gamma_coeff = 1.0 / 2.2

    diffuse_net = KPCN(device, input_channels=num_features).to(device)
    specular_net = KPCN(device, input_channels=num_features).to(device)

    diffuse_net.load_state_dict(torch.load(diffuse_checkpoint_path)['model'])
    specular_net.load_state_dict(torch.load(specular_checkpoint_path)['model'])
    criterion = nn.L1Loss()

    diffuse_net.eval()
    specular_net.eval()
    
    with torch.no_grad():

        criterion = nn.L1Loss()
        
        # make singleton batch
        data = send_to_device(to_torch_tensors(data), device)
        
        if len(data['X_diffuse'].size()) != 4:
            data = unsqueeze_all(data)
        
        print(data['X_diffuse'].size())
        
        # Test Diffuse
        diffuse_data = data['X_diffuse'].permute(permutation).to(device)
        diffuse_data_gt = data['Reference'][:, :, :, :3].permute(permutation).to(device)
        diffuse_output = diffuse_net(diffuse_data)

        # Visualization
        if output_kernel:
            save_kernel(diffuse_output, type='diffuse')
        

        diffuse_output = diffuse_net.apply_kernel(diffuse_output, crop_like(diffuse_data, diffuse_output))
        diffuse_gt = crop_like(diffuse_data_gt, diffuse_output)
        diffuse_loss = criterion(diffuse_output, diffuse_gt).item()

        # Test Speciular
        specular_data = data['X_specular'].permute(permutation).to(device)
        specular_data_gt = data['Reference'][:, :, :, 3:6].permute(permutation).to(device)
        specular_output = specular_net(specular_data)

        # Visualization
        if output_kernel:
            save_kernel(specular_output, type='specular')

        specular_output = specular_net.apply_kernel(specular_output, crop_like(specular_data, specular_output))
        specular_gt = crop_like(specular_data_gt, specular_output)
        specular_loss = criterion(specular_output, specular_gt).item()

        # calculate final ground truth error
        albedo = data['origAlbedo'].permute(permutation).to(device)
        albedo = crop_like(albedo, diffuse_output)
        output = diffuse_output * (albedo + eps) + torch.exp(specular_output) - 1.0

        original = crop_like(data['finalInput'].permute(permutation), output)
        gt = crop_like(data['finalGt'].permute(permutation), output)
        loss = criterion(output, gt).item()
        rmse = RMSE(output, gt)
        
        original = original.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        img = output.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        simm = SSIM(img, gt, True)
        
        img = cv2.cvtColor((img.clip(0, np.max(img)) ** gamma_coeff) * 255., cv2.COLOR_BGR2RGB)
        if output_kernel:
            for i in range(3):
                img = cv2.circle(img, (visual_x[i], visual_y[i]), 5, (0, 0, 255), -1)

        cv2.imwrite(f'test_result/test_{scene_name}.png', cv2.cvtColor((original.clip(0, np.max(original)) ** gamma_coeff) * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'test_result/test_{scene_name}_gt.png', cv2.cvtColor((gt.clip(0, np.max(gt)) ** gamma_coeff) * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'test_result/test_{scene_name}_pred.png', img)
        
        print(f"Diffuse Lose: {diffuse_loss}")
        print(f"Specular Loss: {specular_loss}")
        print(f"Total Loss {loss}")
        print(f"RMSE: {rmse}")
        print(f"SSIM: {simm}")

if __name__ == "__main__":
    
    is_debug = False
    scene_name = 'coffee'
    
    data = preprocess_input(f"data/test/{scene_name}_0.exr", f"data/test/{scene_name}_gt_0.exr", debug=is_debug)
    test(data, scene_name, 'checkpoints/diffuse_checkpoint_63.pth.tar', 'checkpoints/specular_checkpoint_63.pth.tar')