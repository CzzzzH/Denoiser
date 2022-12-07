import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
import pyexr
import os
import argparse

from tqdm import tqdm
from scene_parser import get_shapenet_scene

suffix = '_test'
# suffix = ''

def get_scene_parameters(scene, show_params=False):
    
    params = mi.traverse(scene)
    if show_params:
        print(params)
        for key in params.keys():
            print(f'{key}: {params[key]}')  
    return params

def generate_translation(sample_num, seed):

    np.random.seed(seed)
    translation_vector = np.random.randn(3, sample_num)
    rotation_axis = np.random.randn(3, sample_num)
    translation_vector /= np.linalg.norm(translation_vector, axis=0)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=0)
    return translation_vector.T, rotation_axis.T

def camera_translation(params, translate_vector, rotation_axis, translation_step, rotation_step):
    
    translate_matrix = mi.Transform4f.translate(translate_vector * translation_step)
    rotation_matrix = mi.Transform4f.rotate(axis=rotation_axis, angle=rotation_step)
    params["PerspectiveCamera.to_world"] = translate_matrix @ rotation_matrix @ params["PerspectiveCamera.to_world"]
    params.update()

def render_thumbnail(scene, params, file_name):

    params["PerspectiveCamera.film.size"] //= 2
    params["PerspectiveCamera.film.crop_size"] //= 2
    params.update()
    
    img = mi.render(scene, params=params, spp=8)
    mi.util.write_bitmap(f"thumbnail/{file_name}.png", img[:, :, :3], write_async=False)

    params["PerspectiveCamera.film.size"] *= 2
    params["PerspectiveCamera.film.crop_size"] *= 2
    params.update()

    return img

def write_exr(exr_list, file_name):

    for i in range(len(exr_list)):
        pyexr.write(f"original_kpcn/data/test/{file_name}_{i}.exr", exr_list[i])

def render_avo(scene, sample_num, file_name, gt=False):

    img = scene.integrator().render(scene, spp=sample_num)
    depth = np.array(img[:, :, 7])
    normal = np.array(img[:, :, 8:11])
    position = np.array(img[:, :, 11:14])
    albedo = np.array(img[:, :, 14:17])
    color = np.array(img[:, :, 0:3])
    exr_dict = {'default': color, 'Color': color, 'Depth': depth, 'Normal': normal, 'Position': position, 'Albedo': albedo}
    
    if gt:
        mi.util.write_bitmap(f"original_kpcn/data_vis/{file_name}_gt.png", color, write_async=False)

    return exr_dict

def load_scene(name, h=None, w=None, scene_dict=None, scene_dict_specular=None):

    if scene_dict == None: scene = mi.load_file(f"assets{suffix}/kpcn/{name}/scene_kpcn.xml")
    else: scene = mi.load_dict(scene_dict)
    if scene_dict_specular == None: scene_specular = mi.load_file(f"assets{suffix}/kpcn/{name}/scene_specular.xml")
    else: scene_specular = mi.load_dict(scene_dict_specular)

    params = get_scene_parameters(scene, False)
    params_specular = get_scene_parameters(scene_specular, False)

    if h != None:
        params["PerspectiveCamera.film.size"] = [w, h]
        params["PerspectiveCamera.film.crop_size"] = [w, h]
        params_specular["PerspectiveCamera.film.size"] = [w, h]
        params_specular["PerspectiveCamera.film.crop_size"] = [w, h]
    
    return scene, scene_specular, params, params_specular

def generate_data(name, frame_number, translation_step, rotation_step, minimum_radiance, random_seed, h=None, w=None, scene_dict=None, scene_dict_specular=None):

    exr_list = []
    gt_exr_list = []
    translation_vector, rotation_axis = generate_translation(frame_number, random_seed)

    # Render Scene
    scene, scene_specular, params, params_specular = load_scene(name, h, w, scene_dict, scene_dict_specular) 

    for i in tqdm(range(frame_number)):
        
        if frame_number > 1:
            camera_translation(params, translation_vector[i], rotation_axis[i], translation_step, rotation_step)  
            camera_translation(params_specular, translation_vector[i], rotation_axis[i], translation_step, rotation_step)  
            thumbnail = render_thumbnail(scene, params, f'{name}_{i}')

            if np.array(thumbnail).mean() < minimum_radiance:
                camera_translation(params, translation_vector[i], rotation_axis[i], -3 * translation_step, -rotation_step)  
                camera_translation(params_specular, translation_vector[i], rotation_axis[i], -3 * translation_step, -rotation_step)
                thumbnail = render_thumbnail(scene, params, f'{name}_{i}')

        scene, scene_specular, params, params_specular = load_scene(name, h, w, scene_dict, scene_dict_specular) 
        exr_list.append(render_avo(scene, sample_num=8, file_name=name, gt=False))
        
        scene, scene_specular, params, params_specular = load_scene(name, h, w, scene_dict, scene_dict_specular) 
        gt_exr_list.append(render_avo(scene, sample_num=1000, file_name=f'{name}_{i}', gt=True))
        
        specular_component = scene_specular.integrator().render(scene_specular, spp=8)
        
        scene, scene_specular, params, params_specular = load_scene(name, h, w, scene_dict, scene_dict_specular) 
        gt_specular_component = scene_specular.integrator().render(scene_specular, spp=1000)
        exr_list[i]['Specular'] = np.array(specular_component)
        exr_list[i]['Diffuse'] = exr_list[i]['Color'] - exr_list[i]['Specular']
        gt_exr_list[i]['Specular'] = np.array(gt_specular_component)
        gt_exr_list[i]['Diffuse'] = gt_exr_list[i]['Color'] - gt_exr_list[i]['Specular']

    write_exr(exr_list, name)
    write_exr(gt_exr_list, f'{name}_gt')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-sample', type=int, default=8)
    parser.add_argument('--frame-num', type=int, default=10)
    parser.add_argument('--object-num', type=int, default=500)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--scene-type', type=str, default="indoor")
    args = parser.parse_args()
    
    h = 512
    w = 512
    
    # scene_dict, scene_dict_specular, _ = get_shapenet_dov(h, w)
    # generate_data(f'shapenet_dof', frame_number=1, translation_step=0.5, rotation_step=15, minimum_radiance=1, random_seed=0,
    #                          scene_dict=scene_dict, scene_dict_specular=scene_dict_specular, h=h, w=w)

    # Generate random single / pinhole / depth-of-field shapenet data
    if args.scene_type == "pinhole" or args.scene_type == "dof" or args.scene_type == "single":
        i = 8
        while i <= 10:
            dir = f"assets{suffix}/shapenet"
            print(f"Generating ShapeNet Scene ({args.scene_type}) --> {i}")
            # try:
            scene_dict, scene_dict_specular, _ = get_shapenet_scene(h, w, method='kpcn', scene_type=args.scene_type)
            scene_name = f'shapenet_{args.scene_type}_{i}'
            generate_data(scene_name, frame_number=1, translation_step=0.5, rotation_step=15, minimum_radiance=1, random_seed=i,
                        scene_dict=scene_dict, scene_dict_specular=scene_dict_specular, h=h, w=w)
            i += 1
            # except:
            #     print(f"Error! Retrying...")

    # Genereate indoor & complex data
    if args.scene_type == "indoor" or args.scene_type == "complex":
        asset_dir = os.listdir(f"assets{suffix}/kpcn")
        
        for index in range(len(asset_dir)):
            print(f"Generate Scene {asset_dir[index]}")
            scene_name = asset_dir[index]
            generate_data(scene_name, frame_number=1, translation_step=0.1, rotation_step=5, minimum_radiance=1, random_seed=0, h=None, w=None)
