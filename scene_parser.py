import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
import os

from scene_template import single_plastic, single_plastic_s, single_plastic_gt
from scene_template import pinhole, pinhole_s, pinhole_gt
from scene_template import depth_of_field, depth_of_field_s, depth_of_field_gt

def generate_transform(translation_step, rotation_step):
    
    translation_vector = np.random.uniform(-1, 1, 3)
    rotation_axis = np.random.uniform(-1, 1, 3)
    translation_vector /= np.linalg.norm(translation_vector, axis=0)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=0)
    translate_matrix = mi.ScalarTransform4f.translate(translation_vector * translation_step)
    rotation_matrix = mi.ScalarTransform4f.rotate(axis=rotation_axis, angle=rotation_step)
    transform = translate_matrix @ rotation_matrix
    return transform

def get_shapenet_scene(h, w, method='kpcn', scene_type='pinhole'):

    dir_path = 'assets/shapenet'
    shape_net_list = os.listdir(dir_path)
    shape_net_list.sort()

    object_num = 1 if scene_type == 'single' else 3
    obj_indexes = np.random.randint(0, len(shape_net_list), object_num)

    if scene_type == 'pinhole':
        scene_dict = pinhole
        scene_dict_specular = pinhole_s
        scene_dict_gt = pinhole_gt
    elif scene_type == 'dof':
        scene_dict = depth_of_field
        scene_dict_specular = depth_of_field_s
        scene_dict_gt = depth_of_field_gt
    elif scene_type == 'single':
        scene_dict = single_plastic
        scene_dict_specular = single_plastic_s
        scene_dict_gt = single_plastic_gt
    
    if method == 'kpcn':
        scene_dict['integrator']['aovs'] = 'depth:depth, normal:sh_normal, position:position, albedo:albedo'
        scene_dict['integrator']['path_tracer']['rr_depth'] = 5
        scene_dict_specular['integrator']['rr_depth'] = 5

    # Modify resolution
    scene_dict['PerspectiveCamera']['film']['width'] = w
    scene_dict['PerspectiveCamera']['film']['height'] = h
    scene_dict_specular['PerspectiveCamera']['film']['width'] = w
    scene_dict_specular['PerspectiveCamera']['film']['height'] = h
    scene_dict_gt['PerspectiveCamera']['film']['width'] = w
    scene_dict_gt['PerspectiveCamera']['film']['height'] = h
    
    # Modify camera transform
    if scene_type != 'single':
        camera_transform = generate_transform(0.5, 10)
        scene_dict['PerspectiveCamera']['to_world'] = camera_transform @ scene_dict['PerspectiveCamera']['to_world']
        scene_dict_specular['PerspectiveCamera']['to_world'] = camera_transform @ scene_dict_specular['PerspectiveCamera']['to_world']
        scene_dict_gt['PerspectiveCamera']['to_world'] = camera_transform @ scene_dict_gt['PerspectiveCamera']['to_world']
    
    # Modify environment map type
    if scene_type != 'single':
        env_type = np.random.randint(0, 20)
    else: env_type = 0
    
    scene_dict['EnvironmentMapEmitter']['filename'] = f'D:/UCSD/CSE_274_Graphics/Denoiser/assets/envmaps/envmap_{env_type}.hdr'
    scene_dict_specular['EnvironmentMapEmitter']['filename'] = f'D:/UCSD/CSE_274_Graphics/Denoiser/assets/envmaps/envmap_{env_type}.hdr'
    scene_dict_gt['EnvironmentMapEmitter']['filename'] =  f'D:/UCSD/CSE_274_Graphics/Denoiser/assets/envmaps/envmap_{env_type}.hdr'
    
    # Modify depth of field parameters
    if scene_type != 'single':
        if 'focus_distance' in scene_dict['PerspectiveCamera']:
            focus_distance_var = np.random.uniform(-5, 5)
            aperture_radius_var = np.random.uniform(-0.5, 0.5)
            scene_dict['PerspectiveCamera']['focus_distance'] += focus_distance_var
            scene_dict_specular['PerspectiveCamera']['focus_distance'] += focus_distance_var
            scene_dict_gt['PerspectiveCamera']['focus_distance'] += focus_distance_var
            scene_dict['PerspectiveCamera']['aperture_radius'] += aperture_radius_var
            scene_dict_specular['PerspectiveCamera']['aperture_radius'] += aperture_radius_var
            scene_dict_gt['PerspectiveCamera']['aperture_radius'] += aperture_radius_var
        else:
            fov = 45
            fov_var = np.random.uniform(-15, 15)
            scene_dict['PerspectiveCamera']['fov'] = fov + fov_var
            scene_dict_specular['PerspectiveCamera']['fov'] = fov + fov_var
            scene_dict_gt['PerspectiveCamera']['fov'] = fov + fov_var
    
    else:
        fov = 30
        scene_dict['PerspectiveCamera']['fov'] = fov
        scene_dict_specular['PerspectiveCamera']['fov'] = fov
        scene_dict_gt['PerspectiveCamera']['fov'] = fov
        
    for (i, obj_index) in enumerate(obj_indexes):

        obj_path = os.path.join(dir_path, shape_net_list[obj_index], "models/model_normalized.obj")

        scene_dict[f'Object{i}']['filename'] = obj_path
        scene_dict_specular[f'Object{i}']['filename'] = obj_path
        scene_dict_gt[f'Object{i}']['filename'] = obj_path

        random_translate = np.random.uniform(-1, 1, 3).tolist()
        random_translate[0] = -0.8 + i * 0.8
        random_translate[1] = 0.3
        random_translate[2] *= 0.5
        random_angle = np.random.uniform(0, 360)

        object_scale = 10
        scene_dict[f'Object{i}']['to_world'] = mi.ScalarTransform4f.scale([object_scale, object_scale, object_scale]) @ (scene_dict[f'Object{i}']['to_world']).translate(random_translate).rotate(axis=[0, 1, 0], angle=random_angle)
        scene_dict_specular[f'Object{i}']['to_world'] = mi.ScalarTransform4f.scale([object_scale, object_scale, object_scale]) @ (scene_dict_specular[f'Object{i}']['to_world']).translate(random_translate).rotate(axis=[0, 1, 0], angle=random_angle)
        scene_dict_gt[f'Object{i}']['to_world'] = mi.ScalarTransform4f.scale([object_scale, object_scale, object_scale]) @ (scene_dict_gt[f'Object{i}']['to_world']).translate(random_translate).rotate(axis=[0, 1, 0], angle=random_angle)

        if 'diffuse_reflectance' in scene_dict[f'BSDF{i}']['material'].keys():
            diffuse_reflectance = np.random.rand(3).tolist()
            scene_dict[f'BSDF{i}']['material']['diffuse_reflectance']['value'] = diffuse_reflectance
            scene_dict_specular[f'BSDF{i}']['material']['diffuse_reflectance']['value'] = diffuse_reflectance
            scene_dict_gt[f'BSDF{i}']['material']['diffuse_reflectance']['value'] = diffuse_reflectance
        if 'int_ior' in scene_dict[f'BSDF{i}']['material'].keys():
            int_ior = np.random.rand(1)[0] + 1
            scene_dict[f'BSDF{i}']['material']['int_ior'] = int_ior
            scene_dict_specular[f'BSDF{i}']['material']['int_ior'] = int_ior
            scene_dict_gt[f'BSDF{i}']['material']['int_ior'] = int_ior
        if 'specular_reflectance' in scene_dict[f'BSDF{i}']['material'].keys():
            specular_reflectance = 1 + np.random.rand(1)[0] * 0.05
            scene_dict[f'BSDF{i}']['material']['specular_reflectance'] = specular_reflectance
            scene_dict_specular[f'BSDF{i}']['material']['specular_reflectance'] = specular_reflectance
            scene_dict_gt[f'BSDF{i}']['material']['specular_reflectance'] = specular_reflectance
        if 'specular_transmittance' in scene_dict[f'BSDF{i}']['material'].keys():
            specular_transmittance = 1 + np.random.rand(1)[0] * 0.05
            scene_dict[f'BSDF{i}']['material']['specular_transmittance'] = specular_transmittance
            scene_dict_specular[f'BSDF{i}']['material']['specular_transmittance'] = specular_transmittance
            scene_dict_gt[f'BSDF{i}']['material']['specular_transmittance'] = specular_transmittance
        if 'alpha_u' in scene_dict[f'BSDF{i}']['material'].keys():
            alpha_u = np.random.rand(1)[0] * 0.5
            scene_dict[f'BSDF{i}']['material']['alpha_u'] = alpha_u
            scene_dict_specular[f'BSDF{i}']['material']['alpha_u'] = alpha_u
            scene_dict_gt[f'BSDF{i}']['material']['alpha_u'] = alpha_u
        if 'alpha_v' in scene_dict[f'BSDF{i}']['material'].keys():
            alpha_v = np.random.rand(1)[0] * 0.5
            scene_dict[f'BSDF{i}']['material']['alpha_v'] = alpha_v
            scene_dict_specular[f'BSDF{i}']['material']['alpha_v'] = alpha_v
            scene_dict_gt[f'BSDF{i}']['material']['alpha_v'] = alpha_v
        
    return scene_dict, scene_dict_specular, scene_dict_gt

if __name__ == "__main__":

    scene_dict, scene_dict_specular, scene_dict_gt = get_shapenet_scene(512, 512)
    scene = mi.load_dict(scene_dict)
    scene_specular = mi.load_dict(scene_dict_specular)
    scene_gt = mi.load_dict(scene_dict_gt)

    img = mi.render(scene, spp=32)
    img_specular = mi.render(scene_specular, spp=32)
    img_gt = mi.render(scene_gt, spp=1024)
    mi.util.write_bitmap(f"debug_vis/dov_template.png", img[:, :, :3], write_async=False)
    mi.util.write_bitmap(f"debug_vis/dov_template_specular.png", img_specular, write_async=False)   
    mi.util.write_bitmap(f"debug_vis/dov_template_gt.png", img_gt, write_async=False)
    