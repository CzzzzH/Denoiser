import os
import shutil
import random

def select_data():
    
    shapehet_dir = '/mnt/d/UCSD/dataset/ShapeNetCore.v2'
    class_list = os.listdir(shapehet_dir)
    num_per_class = 56
    
    for object_class in class_list:
        
        class_dir = os.path.join(shapehet_dir, object_class)
        object_list = os.listdir(class_dir)
        random.shuffle(object_list)
        object_list = object_list[:num_per_class]
        
        for object in object_list:
            object_dir = os.path.join(class_dir, object)
            shutil.copytree(object_dir, f'assets/shapenet/{object}')

def generate_data():

    scene_type = 'pinhole'
    data_num = 5
    test = True
    data_dir = 'test' if test == True else 'train'
    
    i = 1
    while i <= data_num:
        if not os.path.exists(f'data/{data_dir}/shapenet_{scene_type}_{i}_data.npz'):
            os.system(f'python generate_data.py --object-index {i} --scene-type {scene_type} {"" if test == False else "--test"}')
            i = 0
        i += 1

if __name__ == '__main__':
    # select_data()
    generate_data()