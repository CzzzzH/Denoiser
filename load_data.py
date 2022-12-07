import cv2
import numpy as np
import os

def output_max_min(output):
    print("Max / Min")
    print(output.shape)
    print("Specular Red")
    print(np.max(np.array(output[:, :, 0])), np.min(np.array(output[:, :, 0])))
    print("Specular Green")
    print(np.max(np.array(output[:, :, 1])), np.min(np.array(output[:, :, 1])))
    print("Specular Blue")
    print(np.max(np.array(output[:, :, 2])), np.min(np.array(output[:, :, 2])))
    print("Diffuse Red")
    print(np.max(np.array(output[:, :, 3])), np.min(np.array(output[:, :, 3])))
    print("Diffuse Green")
    print(np.max(np.array(output[:, :, 4])), np.min(np.array(output[:, :, 4])))
    print("Diffuse Blue")
    print(np.max(np.array(output[:, :, 5])), np.min(np.array(output[:, :, 5])))
    print("Alpha")
    print(np.max(np.array(output[:, :, 6])), np.min(np.array(output[:, :, 6])))
    print("Depth")
    print(np.max(np.array(output[:, :, 7])), np.min(np.array(output[:, :, 7])))
    print("Normal R")
    print(np.max(np.array(output[:, :, 8])), np.min(np.array(output[:, :, 8])))
    print("Normal G")
    print(np.max(np.array(output[:, :, 9])), np.min(np.array(output[:, :, 9])))
    print("Normal B")
    print(np.max(np.array(output[:, :, 10])), np.min(np.array(output[:, :, 10])))
    print("Position R")
    print(np.max(np.array(output[:, :, 11])), np.min(np.array(output[:, :, 11])))
    print("Position G")
    print(np.max(np.array(output[:, :, 12])), np.min(np.array(output[:, :, 12])))
    print("Position B")
    print(np.max(np.array(output[:, :, 13])), np.min(np.array(output[:, :, 13])))
    print("Albedo R")
    print(np.max(np.array(output[:, :, 14])), np.min(np.array(output[:, :, 14])))
    print("Albedo G")
    print(np.max(np.array(output[:, :, 15])), np.min(np.array(output[:, :, 15])))
    print("Albedo B")
    print(np.max(np.array(output[:, :, 16])), np.min(np.array(output[:, :, 16])))
    print("Sample X")
    print(np.max(np.array(output[:, :, 17])), np.min(np.array(output[:, :, 17])))
    print("Sample Y")
    print(np.max(np.array(output[:, :, 18])), np.min(np.array(output[:, :, 18])))
    print("Sample U")
    print(np.max(np.array(output[:, :, 19])), np.min(np.array(output[:, :, 19])))
    print("Sample V")
    print(np.max(np.array(output[:, :, 20])), np.min(np.array(output[:, :, 20])))
    print("Sample Time")
    print(np.max(np.array(output[:, :, 21])), np.min(np.array(output[:, :, 21])))

    for depth in range(5):
        print(f'[{depth + 1}] Diffuse Material')
        print(np.max(np.array(output[:, :, 22 + depth * 9])), np.min(np.array(output[:, :, 22 + depth * 9])))
        print(f'[{depth + 1}] Reflection Material')
        print(np.max(np.array(output[:, :, 23 + depth * 9])), np.min(np.array(output[:, :, 23 + depth * 9])))
        print(f'[{depth + 1}] Transmission Material')
        print(np.max(np.array(output[:, :, 24 + depth * 9])), np.min(np.array(output[:, :, 24 + depth * 9])))
        print(f'[{depth + 1}] Glossy Material')
        print(np.max(np.array(output[:, :, 25 + depth * 9])), np.min(np.array(output[:, :, 25 + depth * 9])))
        print(f'[{depth + 1}] Specular Material')
        print(np.max(np.array(output[:, :, 26 + depth * 9])), np.min(np.array(output[:, :, 26 + depth * 9])))
        print(f'[{depth + 1}] Direction Theta')
        print(np.max(np.array(output[:, :, 27 + depth * 9])), np.min(np.array(output[:, :, 27 + depth * 9])))
        print(f'[{depth + 1}] Direction Phi')
        print(np.max(np.array(output[:, :, 28 + depth * 9])), np.min(np.array(output[:, :, 28 + depth * 9])))
        print(f'[{depth + 1}] Direct Emission Probability')
        print(np.max(np.array(output[:, :, 29 + depth * 9])), np.min(np.array(output[:, :, 29 + depth * 9])))
        print(f'[{depth + 1}] Emitter Emission Probability')
        print(np.max(np.array(output[:, :, 30 + depth * 9])), np.min(np.array(output[:, :, 30 + depth * 9])))

if __name__ == "__main__":

    # name = 'shapenet_0'
    # name = 'shapenet_pinhole_1'
    name = 'bathroom'
    
    # dir = 'train_small'
    dir = 'test'
    
    file_path = f'data/{dir}/{name}_data.npz'
    # file_path = f'data_sbmc/train/{name}/data.npz'
    data = np.load(file_path)
    global_feature = data['global_features'].astype(np.float32)
    sample_feature = data['sample_features'].astype(np.float32)
    gt = data['gt'].astype(np.float32)
    
    feature_index = np.random.randint(0, sample_feature.shape[0])
    
    print(global_feature)
    print(sample_feature.shape)
    print(np.max(np.array(sample_feature[feature_index,:, :, 0])), np.min(np.array(sample_feature[feature_index,:, :, 0])))
    print(np.max(np.array(sample_feature[feature_index,:, :, 1])), np.min(np.array(sample_feature[feature_index,:, :, 1])))
    print(np.max(np.array(sample_feature[feature_index,:, :, 2])), np.min(np.array(sample_feature[feature_index,:, :, 2])))
    output_max_min(sample_feature[feature_index, :, :, 3:])
    
    os.makedirs(f"visualization_features/{name}", exist_ok=True)
    
    cv2.imwrite(f"visualization_features/{name}/gt.png", cv2.cvtColor((gt ** 0.4545454) * 255., cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"visualization_features/{name}/original.png", cv2.cvtColor((np.mean(sample_feature[:, :, :, 0:3], axis=0) ** 0.4545454) * 255., cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"visualization_features/{name}/diffuse.png", 
                cv2.cvtColor((sample_feature[feature_index, :, :, 6:9].clip(0, np.Inf) ** 0.4545454) * 255., cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"visualization_features/{name}/specular.png", 
                cv2.cvtColor(((sample_feature[feature_index, :, :, 0:3] - sample_feature[feature_index, :, :, 6:9]).clip(0, np.Inf) ** 0.4545454) * 255., cv2.COLOR_BGR2RGB))
    
    cv2.imwrite(f"visualization_features/{name}/depth.png", sample_feature[feature_index, :, :, 10] / np.max(sample_feature[feature_index, :, :, 10]) * 255)
    cv2.imwrite(f"visualization_features/{name}/normal.png", cv2.cvtColor(sample_feature[feature_index, :, :, 11:14] * 255., cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"visualization_features/{name}/position.png", cv2.cvtColor(sample_feature[feature_index, :, :, 14:17] * 255., cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"visualization_features/{name}/albedo.png", cv2.cvtColor(sample_feature[feature_index, :, :, 17:20] * 255., cv2.COLOR_BGR2RGB))
    
    for i in range(5):
        cv2.imwrite(f"visualization_features/{name}/diffuse_path_{i}.png", sample_feature[feature_index, :, :, 25 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/reflection_path_{i}.png", sample_feature[feature_index, :, :, 26 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/transmission_path_{i}.png", sample_feature[feature_index, :, :, 27 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/glossy_path_{i}.png", sample_feature[feature_index, :, :, 28 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/specular_path_{i}.png", sample_feature[feature_index, :, :, 29 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/theta_path_{i}.png", sample_feature[feature_index, :, :, 30 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/phi_path_{i}.png", sample_feature[feature_index, :, :, 31 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/p1_path_{i}.png", sample_feature[feature_index, :, :, 32 + i * 9] * 255.)
        cv2.imwrite(f"visualization_features/{name}/p2_path_{i}.png", sample_feature[feature_index, :, :, 33 + i * 9] * 255.)