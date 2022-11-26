import pyexr
import cv2
import numpy as np
import os

if __name__ == "__main__":

    coeff = 1.0 / 2.2

    # dir_path = '/mnt/d/UCSD课件/CSE_274_Graphics/sbmc/data/renderings/ref'
    dir_path = './data'

    list = os.listdir(dir_path)

    for (i, file_name) in enumerate(list):

        if not 'dov_5_1' in file_name:
            continue

        file_path = os.path.join(dir_path, file_name)
        file = pyexr.open(file_path)

        print("Available channels:")
        file.describe_channels()

        color = file.get("Color")
        diffuse = file.get("Diffuse")
        specular = file.get("Specular")
        depth = file.get("Depth")
        albedo = file.get("Albedo")
        normal = file.get("Normal")
        position = file.get("Position")

        color = np.clip(file.get("Color"), 0, np.max(color)) ** coeff
        diffuse = np.clip(file.get("Diffuse"), 0, np.max(diffuse)) ** coeff
        specular = np.clip(file.get("Specular"), 0, np.max(specular)) ** coeff


        print("Max/Min Color: ", np.max(color), np.min(color))
        print("Max/Min Diffuse: ", np.max(diffuse), np.min(diffuse))
        print("Max/Min Specular: ", np.max(specular), np.min(specular))
        print("Max/Min Albedo: ", np.max(albedo), np.min(albedo))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_color.png", cv2.cvtColor(color * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_depth.png", depth)
        cv2.imwrite(f"data_vis/{file_name[:-4]}_albedo.png", cv2.cvtColor(albedo * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_normal.png", cv2.cvtColor(normal * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_position.png", cv2.cvtColor(position * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_diffuse.png", cv2.cvtColor(diffuse * 255., cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"data_vis/{file_name[:-4]}_specular.png", cv2.cvtColor(specular * 255., cv2.COLOR_BGR2RGB))
