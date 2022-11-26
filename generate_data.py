import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
import os
import time
import argparse
from scene_parser import get_shapenet_scene
        
def test_generate_time(scene_dict, scene_dict_specular, scene_dict_gt):

    print("Start testing generate time...")
    scene = mi.load_dict(scene_dict)
    scene_specular = mi.load_dict(scene_dict_specular)
    scene_gt = mi.load_dict(scene_dict_gt)
    
    time_start = time.time()
    for i in range(5):
        gt = scene_gt.integrator().render(scene_gt, spp=1024)
        print(f'[{i}] Time: {time.time() - time_start}')
    time_end = time.time()
    print(f"Time generate GT: {time_end - time_start}")
    
    time_start = time.time()
    for i in range(10):
        scene = mi.load_dict(scene_dict)
        output = np.array(scene.integrator().render(scene, spp=1, seed=int(time.time())))
        print(f'[{i}] Time: {time.time() - time_start}')
    time_end = time.time()
    print(f"Time generate Input: {time_end - time_start}")

def generate_transform(translation_step, rotation_step):
    
    translation_vector = np.random.uniform(-1, 1, 3)
    rotation_axis = np.random.uniform(-1, 1, 3)
    translation_vector /= np.linalg.norm(translation_vector, axis=0)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=0)
    translate_matrix = mi.Transform4f.translate(translation_vector * translation_step)
    rotation_matrix = mi.Transform4f.rotate(axis=rotation_axis, angle=rotation_step)
    transform = translate_matrix @ rotation_matrix
    return transform

def load_scene(dir, scene_dict=None, scene_dict_specular=None, scene_dict_gt=None, transform=None, test=False, gt=True):
    
    if scene_dict == None: scene = mi.load_file(os.path.join(dir, f"{name}/scene_sbmc.xml"))
    else: scene = mi.load_dict(scene_dict)
    if scene_dict_specular == None: scene_specular = mi.load_file(os.path.join(dir, f"{name}/scene_specular.xml"))
    else: scene_specular = mi.load_dict(scene_dict_specular)
    
    scene_gt = None
    
    if gt:
        if scene_dict_gt == None: scene_gt = mi.load_file(os.path.join(dir, f"{name}/scene_gt.xml"))
        else: scene_gt = mi.load_dict(scene_dict_gt)
        params_gt = mi.traverse(scene_gt)
        
    params = mi.traverse(scene)
    params_specular = mi.traverse(scene_specular)
    
    if test == False:
        params["PerspectiveCamera.film.size"] = [128, 128]
        params["PerspectiveCamera.film.crop_size"] = [128, 128]    
        params_specular["PerspectiveCamera.film.size"] = [128, 128]
        params_specular["PerspectiveCamera.film.crop_size"] = [128, 128]
        
        if gt:
            params_gt["PerspectiveCamera.film.size"] = [128, 128]
            params_gt["PerspectiveCamera.film.crop_size"] = [128, 128]

    if transform != None:
        params["PerspectiveCamera.to_world"] = transform @ params["PerspectiveCamera.to_world"]
        params_specular["PerspectiveCamera.to_world"] = transform @ params_specular["PerspectiveCamera.to_world"]
        
        if gt:
            params_gt["PerspectiveCamera.to_world"] = transform @ params_gt["PerspectiveCamera.to_world"]

    return scene, scene_specular, scene_gt

def generate_data(name, dir, total_sample=8, frame=0, preset_transform=None, scene_dict=None, scene_dict_specular=None, scene_dict_gt=None, test=False):

    gt = np.array([0])
    
    transform = preset_transform
    translation_step = 0.2
    rotation_step = 10
    
    if frame > 0 and transform == None:
        
        while gt.mean() > 1 or gt.std() < 0.1: # A rough test of the quality of the generated image
    
            transform = generate_transform(translation_step, rotation_step)
            scene, scene_specular, scene_gt = load_scene(dir, scene_dict, scene_dict_specular, scene_dict_gt, transform, test)
            gt = np.array(scene_gt.integrator().render(scene_gt, spp=1024))
    
    scene, scene_specular, scene_gt = load_scene(dir, scene_dict, scene_dict_specular, scene_dict_gt, transform, test)
    gt = scene_gt.integrator().render(scene_gt, spp=1024 if test else 8192)
    mi.util.write_bitmap(f"visualization_gt/gt_{name}.png", gt[:, :, :3], write_async=False)
    gt = np.array(gt)
    
    params = mi.traverse(scene)
    global_features = np.array([0., 2 * np.pi * params['PerspectiveCamera.x_fov'][0] / 360., 0.])
    if 'PerspectiveCamera.focus_distance' in params:
        global_features[0] = params['PerspectiveCamera.focus_distance'][0]

    output = None
    output_specular = None

    final_data = np.zeros((total_sample, gt.shape[0], gt.shape[1], 70))
        
    for i in range(total_sample):

        # Reload the scene
        scene, scene_specular, _ = load_scene(dir, scene_dict, scene_dict_specular, scene_dict_gt, transform, test, False)
        
        seed = int(time.time())
        output = np.array(scene.integrator().render(scene, spp=1, seed=seed))
        output_specular = np.array(scene_specular.integrator().render(scene_specular, spp=1, seed=seed))
        final_data[i, :, :, 0:3] = output[:, :, 0:3] # Radiance
        final_data[i, :, :, 3:6] = output_specular # Specular
        final_data[i, :, :, 6:9] = final_data[i, :, :, 0:3] - final_data[i, :, :, 3:6] # Diffuse
        final_data[i, :, :, 9:] = output[:, :, 6:]

    final_data[:, :, :, 3:6] = np.log(final_data[:, :, :, 3:6] + 1) # Adjust Specular
    # final_data[:, :, :, 6:9] = final_data[:, :, :, 6:9].clip(0, np.Inf) # Adjust Diffuse (Not needed)
    
    for depth in range(5): # Adjust Probability
        final_data[:, :, :, 32 + depth * 9] = np.log(final_data[:, :, :, 32 + depth * 9] + 1)
        final_data[:, :, :, 33 + depth * 9] = np.log(final_data[:, :, :, 33 + depth * 9] + 1)

    final_data = final_data.astype(np.float32)
    gt = gt.astype(np.float32)
        
    if test:
        np.savez_compressed(f"data/test/{name}_data.npz", global_features=global_features, sample_features=final_data, gt=gt)
    else:
        np.savez_compressed(f"data/train/{name}_data.npz", global_features=global_features, sample_features=final_data, gt=gt)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-index', type=int, default=1)
    parser.add_argument('--total-sample', type=int, default=8)
    parser.add_argument('--frame-num', type=int, default=10)
    parser.add_argument('--object-num', type=int, default=500)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--scene-type', type=str, default="complex")
    args = parser.parse_args()
    
    suffix = '_test' if args.test else ''
    
    # Generate random single / pinhole / depth-of-field shapenet data
    if args.scene_type == "pinhole" or args.scene_type == "dof" or args.scene_type == "single":
        
        if args.test: h, w = 512, 512
        else: h, w = 128, 128
        
        dir = f"assets{suffix}/shapenet"
        print(f"Generating ShapeNet Scene ({args.scene_type}) --> {args.object_index}")

        scene_dict, scene_dict_specular, scene_dict_gt = get_shapenet_scene(h, w, method='sbmc', scene_type=args.scene_type)
        scene_name = f'shapenet_{args.scene_type}_{args.object_index}'
        generate_data(scene_name, dir, args.total_sample, 0, None, scene_dict, scene_dict_specular, scene_dict_gt, args.test)
        
    # Genereate indoor data
    if args.scene_type == "indoor":
        
        dir = f"assets{suffix}/indoor"
        asset_dir = os.listdir(dir)
        
        for index in range(len(asset_dir)):
            
            name = asset_dir[index]
            
            if args.test:
                test_case = 'living-room'
                if test_case not in name:
                    continue
                print(f"Generate Scene {asset_dir[index]} -> Test")
                generate_data(f'{name}', dir, total_sample=args.total_sample, frame=0, test=args.test)
                
            else:
                
                for i in range(args.frame_num):

                    current_frame = i + 1
                    if os.path.exists(f"data/train/{name}_{current_frame}_data.npz"):
                        continue
                    print(f"Generate Scene {asset_dir[index]} -> Frame {current_frame}")
                    generate_data(f'{name}_{current_frame}', dir, total_sample=args.total_sample, frame=current_frame, test=args.test)
        

    # Genereate complex object data
    if args.scene_type == "complex":
        
        dir = f"assets{suffix}/complex"
        asset_dir = os.listdir(dir)
        
        for index in range(len(asset_dir)):
            
            name = asset_dir[index]
            for i in range(args.frame_num):
                
                current_frame = i + 1
                if os.path.exists(f"data/train/{name}_{current_frame}_data.npz"):
                    continue
                print(f"Generate Scene {asset_dir[index]} -> Frame {current_frame}")
                generate_data(f'{name}_{current_frame}', dir, total_sample=args.total_sample, frame=current_frame, test=args.test)