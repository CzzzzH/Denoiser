from random import weibullvariate
from util import *
import json
import os

def get_batch(list, batch_size):
    return [sum(x) / float(len(x))
        for x in (list[k: k + batch_size]
                for k in range(0, len(list), batch_size))]

def smooth(list, weight=0.6):
    new_list = []
    last = list[0]
    new_list.append(last)
    for i in range(len(list)):
        last = (1 - weight) * list[i] + weight * last
        new_list.append(last)
    return new_list

def get_single_curve(dir_path, file_name, title, batch_size):

    l1_list = []
    l2_list = []
    relative_l1_list = []
    relative_l2_list = []
    tonemapped_relative_l2_list = []
    ssim_list = []
    
    with open(f'statistics/{file_name}.json') as f:
        loss_list = json.load(f)
    
    for data in loss_list:
        l1_list.append(np.log(data['l1']))
        l2_list.append(np.log(data['l2']))
        relative_l1_list.append(np.log(data['l1_relative']))
        relative_l2_list.append(np.log(data['l2_relative']))
        tonemapped_relative_l2_list.append(np.log(data['tonemapped_l2_relative']))
        ssim_list.append(np.log(data['SSIM']))

    # for data in loss_list:
    #     l1_list.append(data['l1'])
    #     l2_list.append(data['l2'])
    #     relative_l1_list.append(data['l1_relative'])
    #     relative_l2_list.append(data['l2_relative'])
    #     tonemapped_relative_l2_list.append(data['tonemapped_l2_relative'])
    # ssim_list.append(data['SSIM'])
    
    # l1_list = get_batch(l1_list, batch_size)
    # l2_list = get_batch(l2_list, batch_size)
    # relative_l1_list = get_batch(relative_l1_list, batch_size)
    # relative_l2_list = get_batch(relative_l2_list, batch_size)
    # tonemapped_relative_l2_list = get_batch(tonemapped_relative_l2_list, batch_size)
    # ssim_list = get_batch(ssim_list, batch_size)

    l1_list = smooth(get_batch(l1_list, batch_size))
    l2_list = smooth(get_batch(l2_list, batch_size))
    relative_l1_list = smooth(get_batch(relative_l1_list, batch_size))
    relative_l2_list = smooth(get_batch(relative_l2_list, batch_size))
    tonemapped_relative_l2_list = smooth(get_batch(tonemapped_relative_l2_list, batch_size))
    ssim_list = smooth(get_batch(ssim_list, batch_size))

    # plot_data_single("Train Loss", len(l1_list) * batch_size, batch_size, "Loss(log)",
    #     [tonemapped_relative_l2_list],
    #     ["tonemapped_relative_L2"], os.path.join(dir_path, "loss.png"))
    
    # plot_data_single("Train Loss", len(l1_list) * batch_size, batch_size, "Loss(log)",
    #     [l1_list],
    #     ["l1"], os.path.join(dir_path, "loss_kpcn.png"))
    
    plot_data_single(f"{title} Train Loss", len(l1_list) * batch_size, batch_size, "Loss(log)",
        [l1_list, l2_list, relative_l1_list, relative_l2_list, tonemapped_relative_l2_list, ssim_list],
        ["L1_loss", "L2_loss", "Relative_L1_loss", "Relative_L2_loss", "Tonemapped_Relative_L2_loss", "SSIM"], os.path.join(dir_path, f"{file_name}.png"))

    # plot_data_single(f"{title} Train Loss", len(l1_list) * batch_size, batch_size, "Loss(log)",
    #     [l1_list, l2_list, relative_l1_list, relative_l2_list, ssim_list],
    #     ["L1_loss", "L2_loss", "Relative_L1_loss", "Relative_L2_loss", "SSIM"], os.path.join(dir_path, f"{file_name}.png"))

def get_compared_curve(dir_path, file_name_1, file_name_2, title_1, title_2, batch_size):
    
    l1_list_1 = []
    tonemapped_relative_l2_list_1 = []
    ssim_list_1 = []
    
    l1_list_2 = []
    tonemapped_relative_l2_list_2 = []
    ssim_list_2 = []
    
    with open(f'statistics/{file_name_1}.json') as f:
        loss_list_1 = json.load(f)
    
    with open(f'statistics/{file_name_2}.json') as f:
        loss_list_2 = json.load(f)
    
    for data in loss_list_1:
        l1_list_1.append(np.log(data['l1']))
        tonemapped_relative_l2_list_1.append(np.log(data['tonemapped_l2_relative']))
        ssim_list_1.append(np.log(data['SSIM']))
    
    for data in loss_list_2:
        l1_list_2.append(np.log(data['l1']))
        tonemapped_relative_l2_list_2.append(np.log(data['tonemapped_l2_relative']))
        ssim_list_2.append(np.log(data['SSIM']))

    l1_list_1 = smooth(get_batch(l1_list_1, batch_size))
    tonemapped_relative_l2_list_1 = smooth(get_batch(tonemapped_relative_l2_list_1, batch_size))
    ssim_list_1 = smooth(get_batch(ssim_list_1, batch_size))
    
    l1_list_2 = smooth(get_batch(l1_list_2, batch_size))
    tonemapped_relative_l2_list_2 = smooth(get_batch(tonemapped_relative_l2_list_2, batch_size))
    ssim_list_2 = smooth(get_batch(ssim_list_2, batch_size))

    plot_data_single("L1 Loss", len(l1_list_1) * batch_size, batch_size, "Loss(log)",
        [l1_list_1, l1_list_2], [title_1, title_2], os.path.join(dir_path, f"{title_1}_{title_2}_l1.png"))
    
    plot_data_single("TonemappedRelativeL2 Loss", len(tonemapped_relative_l2_list_1) * batch_size, batch_size, "Loss(log)",
        [tonemapped_relative_l2_list_1, tonemapped_relative_l2_list_2], [title_1, title_2], os.path.join(dir_path, f"{title_1}_{title_2}_tonemapped.png"))
    
    plot_data_single("SSIM Loss", len(ssim_list_2) * batch_size, batch_size, "Loss(log)",
        [ssim_list_1, ssim_list_2], [title_1, title_2], os.path.join(dir_path, f"{title_1}_{title_2}_ssim.png"))

if __name__ == "__main__":

    dir_path = 'visualization_curves'
    file_name_1 = 'sbmc_loss_1150'
    file_name_2 = 'sbmc_loss_1150'
    batch_size = 500
    
    get_single_curve(dir_path, file_name_1, "SBMC", batch_size)
    # get_compared_curve(dir_path, file_name_1, file_name_2, "KPCN", "SBMC", batch_size)
    
    