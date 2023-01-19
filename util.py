import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='sbmc')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    return args

def show_data_compare(index, data1, data2, figsize=(15, 15)):

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    
    ax1.imshow(data1, aspect='equal')
    ax2.imshow(data2, aspect='equal')
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.savefig(f'visualization_debug/compare_{index}.png')
    plt.close()

def unsqueeze_all(d):

    for k, v in d.items():
        d[k] = torch.unsqueeze(v, dim=0)
    return d

def plot_data(title, max_iter, iter_batch, y_label_left, y_label_right, data_list_left, label_list_left, data_list_right, label_list_right, save_path):

    iters = np.linspace(1, max_iter, max_iter // iter_batch)
    axis = plt.figure().add_subplot()
    plot_list = []
    color_list_left = ['red', 'darkorange', 'yellowgreen']
    color_list_right = ['seagreen', 'royalblue', 'darkviolet']
    for i in range(len(data_list_left)):
        plot_list.append(axis.plot(iters, data_list_left[i], color=color_list_left[i], label=label_list_left[i]))
    axis.set_xlabel('Iters')
    axis.set_ylabel(y_label_left)

    right_axis = axis.twinx()
    right_axis.set_ylabel(y_label_right)
    for i in range(len(data_list_right)):
        plot_list.append(right_axis.plot(iters, data_list_right[i], color=color_list_right[i], label=label_list_right[i]))

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    plt.legend(handles_left + handles_right, labels_left + labels_right, loc='center right')
    plt.title(title)

    plt.savefig(save_path)

def plot_data_single(title, max_iter, iter_batch, y_label, data_list, label_list, save_path, conceal_y_axis=False):

    iters = np.linspace(1, max_iter, max_iter // iter_batch)
    axis = plt.figure().add_subplot()
    color_list_left = ['red', 'darkorange', 'yellowgreen', 'seagreen', 'royalblue', 'darkviolet']
    
    lns = []

    for i in range(len(data_list)):
        new_axis = axis.twinx()
        lns += new_axis.plot(iters, data_list[i], color=color_list_left[i], label=label_list[i])
        new_axis.get_yaxis().set_ticks([])

    lables = [l.get_label() for l in lns]
    axis.legend(lns, lables, loc=0)
    axis.set_xlabel('Iterations')
    axis.set_ylabel(y_label)
    axis.set_facecolor('#eafff5')
    
    if conceal_y_axis:
        axis.get_yaxis().set_ticks([])
    
    axis.set_xlim(xmin=0)
    axis.grid(True)
    
    plt.title(title)
    plt.savefig(save_path)