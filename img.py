import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image, ImageDraw
import os
import random
from pathlib import Path

def obb_to_aabb(obb):
    x_coords = obb[0::2]  # [x1, x2, x3, x4]
    y_coords = obb[1::2]  # [y1, y2, y3, y4]
    x_min, x_max = torch.min(x_coords), torch.max(x_coords)
    y_min, y_max = torch.min(y_coords), torch.max(y_coords)
    return torch.tensor([x_min, y_min, x_max, y_max])

class ImageLabelDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.image_paths, self.label_paths = [], []
        self.transform = transform
        for root_dir in root_dirs:
            img_dir, lbl_dir = os.path.join(root_dir, 'images'), os.path.join(root_dir, 'labels')
            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, img_file.replace('.jpg', '.txt'))
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    self.image_paths.append(img_path)
                    self.label_paths.append(lbl_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label_path = self.label_paths[idx]
        obb_list = []
        with open(label_path, 'r') as f:
            # Read all the lines, each line represents a label with coordinates
            for line in f.readlines():
                label_split = line.strip().split()[1:]  # Skip the class number
                obb_coords = torch.tensor([float(coord) for coord in label_split])
                obb_list.append(obb_coords)

        image = self.transform(image) if self.transform else T.ToTensor()(image)
        c, h, w = image.shape

        # Process both OBB and AABB for all bounding boxes
        aabb_list = []
        for obb_coords in obb_list:
            aabb = obb_to_aabb(obb_coords)
            aabb[[0, 2]] *= w
            aabb[[1, 3]] *= h
            obb_coords[0::2] *= w  # Scale x coordinates
            obb_coords[1::2] *= h  # Scale y coordinates
            aabb_list.append((obb_coords, aabb))

        return image, aabb_list

import random
import matplotlib.pyplot as plt
import numpy as np
import os

def show_random_image_with_bbox(dataset, output_dir='output', file_name='test.png'):
    idx = random.randint(0, len(dataset) - 1)
    image, aabb_list = dataset[idx]

    # Convert the image tensor to a numpy array
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Scale pixel values from [-1, 1] back to [0, 1] for displaying
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    # Draw all bounding boxes (AABB and OBB)
    for obb_coords, aabb in aabb_list:
        # Draw the AABB
        rect = plt.Rectangle(
            (aabb[0], aabb[1]),  # (x_min, y_min)
            aabb[2] - aabb[0],   # width (x_max - x_min)
            aabb[3] - aabb[1],   # height (y_max - y_min)
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Draw the OBB by connecting the points with a blue line
        obb_points = obb_coords.view(4, 2).cpu().numpy()
        obb_polygon = plt.Polygon(obb_points, closed=True, edgecolor='blue', fill=None, linewidth=2)
        ax.add_patch(obb_polygon)

    plt.axis('off')  # Hide axis

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure to a file
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Image with AABBs and OBBs saved at {output_path}")

# Paths to data directories (update with your paths)
paths = [Path("../data/nondiff_multi_dense")]
transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

dataset = ImageLabelDataset(root_dirs=paths, transform=transform)

# Show random image with bounding boxes
show_random_image_with_bbox(dataset)
