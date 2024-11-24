import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageRegressionDataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        image_dir, 
        transform=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        self.pairs = []
        self.image_folders = [f for f in os.listdir(image_dir)]

        for img_folder in self.image_folders:
            base_name = img_folder
            try:
                label_row = self.data[self.data['Design'] == base_name]
                if not label_row.empty:
                    label = label_row['Average Cd'].values[0]
                    self.pairs.append((img_folder, label))
                else:
                    pass
                    # print(f"Warning: Label for folder {img_folder} not found in CSV.")
            except Exception as e:
                pass
                # print(f"Error processing {img_folder}: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_folder, label = self.pairs[idx]
        
        img_path = os.path.join(self.image_dir, img_folder)
        
        images = []
        for img in glob.glob(img_path + '/*.png'):
            if 'spherical' in img or 'cylinder' in img:
                images.append(Image.open(img).convert('RGB'))
            elif 'up' in img or 'left' in img or 'front' in img:
                images.append(Image.open(img).convert('L'))
            else: continue
        
        if images is None:
            raise ValueError(f"Error loading image: {img_path}")

        if self.transform:
            images = [self.transform(img) for img in images]


        return torch.cat(images, dim = 0), torch.tensor(label, dtype=torch.float32)
