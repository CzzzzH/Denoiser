import cv2
import os

if __name__ == "__main__":

    detail_start = [90, 170]
    detail_end = [220, 300]

    dir_path = 'experiments_result/finetune_complex'
    img_names = os.listdir(dir_path)
    filter = 'none'
    
    for img_name in img_names:
        
        if filter in img_name:
            continue
                
        path = os.path.join(dir_path, img_name)
        img = cv2.imread(path)
        
        if 'kpcn' in img_name:
            detail_img = img[(detail_start[1] - 8):(detail_end[1] - 8), (detail_start[0] - 8):(detail_end[0] - 8)]
        else:
            detail_img = img[detail_start[1]:detail_end[1], detail_start[0]:detail_end[0]]
        cv2.imwrite(f'{dir_path}/{img_name[:-4]}_detail.png', detail_img)
        
        if 'kpcn' in img_name:
            new_detail_start = [ detail_start[0] - 8, detail_start[1] - 8]
            new_detail_end = [ detail_end[0] - 8, detail_end[1] - 8]
            new_img = cv2.rectangle(img, new_detail_start, new_detail_end, (0, 0, 255), thickness=3)
        else:
            new_img = cv2.rectangle(img, detail_start, detail_end, (0, 0, 255), thickness=3)
        cv2.imwrite(f'{dir_path}/{img_name[:-4]}_new.png', new_img)
        