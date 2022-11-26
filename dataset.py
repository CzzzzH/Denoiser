import torch
from tqdm import tqdm
import numpy as np

class KPCNDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list):
        self.data_list = data_list
        self.eps = 1e-2

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        data = self.data_list[idx]
        sample_features = data["sample_features"]
        gt = data["gt"]

        sample = {}
        sample["radiance"] = torch.Tensor(sample_features[:, :, :, :3]).permute(0, 3, 1, 2)
        sample["kpcn_albedo"] = torch.Tensor(sample_features[:, :, :, 17:20])
        sample["kpcn_albedo"] = torch.mean(sample["kpcn_albedo"], dim=0).permute(2, 0, 1)
        sample["kpcn_diffuse_in"] = torch.Tensor(sample_features[:, :, :, 6:])
        sample["kpcn_diffuse_in"] = torch.mean(sample["kpcn_diffuse_in"], dim=0).permute(2, 0, 1)
        sample["kpcn_diffuse_in"][:3, :, :] /= (sample["kpcn_albedo"] + self.eps)
        sample["kpcn_specular_in"] = torch.cat([torch.Tensor(sample_features[:, :, :, 3:6]), torch.Tensor(sample_features[:, :, :, 9:])], dim=3)
        sample["kpcn_specular_in"] = torch.mean(sample["kpcn_specular_in"], dim=0).permute(2, 0, 1)
        sample["kpcn_diffuse_buffer"] = sample["kpcn_diffuse_in"][:3, :, :].clone()
        sample["kpcn_specular_buffer"] = sample["kpcn_specular_in"][:3, :, :].clone()
        gt = torch.Tensor(gt)
        
        return sample, gt

class SBMCDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list):
        self.data_list = data_list
        
        data_example = data_list[0]
        self.h, self.w = data_example["gt"].shape[:2]
        self.global_features = torch.Tensor(data_example["global_features"]).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        data = self.data_list[idx]
        sample_features = data["sample_features"]
        gt = data["gt"]

        sample = {}
        sample["radiance"] = torch.Tensor(sample_features[:, :, :, :3]).permute(0, 3, 1, 2)
        sample["features"] = torch.Tensor(sample_features[:, :, :, 3:]).permute(0, 3, 1, 2)
        sample["global_features"] = self.global_features
        gt = torch.Tensor(gt)
        
        return sample, gt
