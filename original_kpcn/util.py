import torch
import gc
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2

def show_data(title, data, figsize=(15, 15), normalize=False):

    if normalize:
        data = np.clip(data, 0, 1) ** 0.45454545

    cv2.imwrite(f'test_result/{title}.png', cv2.cvtColor(data * 255., cv2.COLOR_BGR2RGB))

def show_data_sbs(index, data1, data2, figsize=(15, 15)):

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    
    ax1.imshow(data1, aspect='equal')
    ax2.imshow(data2, aspect='equal')
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.savefig(f'debug_vis/compare_{index}.png')
    plt.close()

def to_torch_tensors(data):

    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                data[k] = torch.from_numpy(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if not isinstance(v, torch.Tensor):
                data[i] = to_torch_tensors(v)
        
    return data
    
def send_to_device(data, device):

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = v.to(device)
    
    return data

def getsize(obj):

    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object reffered to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

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

def plot_data_single(title, max_iter, iter_batch, y_label, data_list, label_list, save_path):

    iters = np.linspace(1, max_iter, max_iter // iter_batch)
    axis = plt.figure().add_subplot()
    plot_list = []
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
    axis.get_yaxis().set_ticks([])
    axis.set_xlim(xmin=0)
    axis.grid(True)

    plt.title(title)
    plt.savefig(save_path)