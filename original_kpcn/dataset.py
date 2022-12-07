import torch

class KPCNDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        
        return len(self.data_list)

    def __getitem__(self, idx):
        
        data = self.data_list[idx]
        return data