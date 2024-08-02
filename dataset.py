import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}


    if dataset_name == 'Visa':
        obj_list = ['pcb1', 'pcb2', 'pcb3', 'pcb4']
    elif dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        print(f"Class names in meta_info: {self.cls_names}")  # Debugging statement

        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        print(f"Generated class_name_map_class_id: {self.class_name_map_class_id}")  # Debugging statement

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
        print(f"Accessing class name: {cls_name}")  # Debugging statement

        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        cls_id = self.class_name_map_class_id.get(cls_name.strip(), -1)  # Use .get() to avoid KeyError
        print(f"Class ID for {cls_name}: {cls_id}")  # Debugging statement
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path), "cls_id": cls_id}
