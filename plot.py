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

if __name__ == "__main__":

    dir_path = 'visualization_curves'
    bs = 50
    
    l1_list = []
    l2_list = []
    relative_l1_list = []
    relative_l2_list = []
    tonemapped_relative_l2_list = []
    ssim_list = []

    with open('statistics/kpcn_loss_1020.json') as f:
        loss_list = json.load(f)
    
    # loss_list = loss_list[6000:]
    
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
    #     ssim_list.append(data['SSIM'])
    
    l1_list = smooth(get_batch(l1_list, bs))
    l2_list = smooth(get_batch(l2_list, bs))
    relative_l1_list = smooth(get_batch(relative_l1_list, bs))
    relative_l2_list = smooth(get_batch(relative_l2_list, bs))
    tonemapped_relative_l2_list = smooth(get_batch(tonemapped_relative_l2_list, bs))
    ssim_list = smooth(get_batch(ssim_list, bs))

    # plot_data_single("Train Loss", len(l1_list) * bs, bs, "Loss(log)",
    #     [tonemapped_relative_l2_list],
    #     ["tonemapped_relative_L2"], os.path.join(dir_path, "loss.png"))
    
    # plot_data_single("Train Loss", len(l1_list) * bs, bs, "Loss(log)",
    #     [l1_list],
    #     ["l1"], os.path.join(dir_path, "loss_kpcn.png"))
    
    plot_data_single("Train Loss", len(l1_list) * bs, bs, "Loss(log)",
        [l1_list, l2_list, relative_l1_list, relative_l2_list, tonemapped_relative_l2_list, ssim_list],
        ["L1_loss", "L2_loss", "Relative_L1_loss", "Relative_L2_loss", "Tonemapped_Relative_L2_loss", "SSIM"], os.path.join(dir_path, "loss_kpcn.png"))