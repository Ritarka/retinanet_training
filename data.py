import torch
import torchvision
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import os
import gc
from pathlib import Path

def obb_to_aabb(obb):
    """
    Converts an Oriented Bounding Box (OBB) defined by 4 points into
    an Axis-Aligned Bounding Box (AABB).

    :param obb: Tensor of shape (8,) representing 4 corners of the OBB 
                in the format [x1, y1, x2, y2, x3, y3, x4, y4].
    :return: Tensor of shape (4,) representing the AABB in the format 
             [x_min, y_min, x_max, y_max].
    """
    # Extract x and y coordinates
    x_coords = obb[0::2]  # [x1, x2, x3, x4]
    y_coords = obb[1::2]  # [y1, y2, y3, y4]

    # Compute the minimum and maximum coordinates
    x_min = torch.min(x_coords)
    y_min = torch.min(y_coords)
    x_max = torch.max(x_coords)
    y_max = torch.max(y_coords)

    return torch.tensor([x_min, y_min, x_max, y_max])


class ImageLabelDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Custom Dataset to load images and labels from multiple directories.

        :param root_dirs: List of directories that contain "images" and "labels" subdirectories.
        :param transform: Optional transforms to apply to the images.
        """
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

        # Collect all image and label file paths
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                raise Exception("dir does not exist " + str(root_dir))
            image_dir = os.path.join(root_dir, 'images')
            label_dir = os.path.join(root_dir, 'labels')

            for img_file in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_file)
                label_file = img_file.replace('.jpg', '.txt')  # Assuming label files are .txt; adjust if needed
                label_path = os.path.join(label_dir, label_file)

                # Check if both image and label exist
                if os.path.exists(img_path) and os.path.exists(label_path):
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

        # Load label
        label_path = self.label_paths[idx]
            
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                # Example format: "0 x1 y1 x2 y2 x3 y3 x4 y4"
                label_split = line.strip().split()
                class_number = 1  # Class number (first value)
                obb_coords = torch.tensor([float(coord) for coord in label_split[1:]])  # Extract the 8 coordinates

                # Convert OBB to AABB
                aabb = obb_to_aabb(obb_coords)  # Function we defined earlier
                boxes.append(aabb)
                labels.append(class_number)

        # Convert to tensors
        boxes = torch.stack(boxes)  # Shape: (N, 4) for N boxes
        labels = torch.tensor(labels, dtype=torch.int64)  # Shape: (N,)

        # Apply transformations to the image if necessary
        image = self.transform(image)
        c, h, w = image.shape

        # Adjust AABB coordinates to the image size
        boxes[:, [0, 2]] *= w  # Scale x coordinates (x_min, x_max)
        boxes[:, [1, 3]] *= h  # Scale y coordinates (y_min, y_max)

        # Prepare the target in the format the model expects:
        # - boxes (FloatTensor[N, 4]): ground-truth boxes in [x1, y1, x2, y2]
        # - labels (Int64Tensor[N]): the class label for each ground-truth box
        target = {
            'boxes': boxes,  # Shape: (N, 4) for N boxes
            'labels': labels  # Shape: (N,)
        }

        return image, target



def get_dataloaders(
    dataset_name: str = "mnist",
    batch_size: int = 32,
    train_transforms=None,
    test_transforms=None,
):
    # usually, we want to split data into training, validation, and test sets
    # for simplicity, we will only use training and test sets
    if train_transforms is None:
        train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    if test_transforms is None:
        test_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    if dataset_name == "mnist":
        # Do Not Change This Code
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transforms
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transforms
        )

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "cifar100":
        # TODO: Load CIFAR100 dataset
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transforms)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transforms)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "products":
        paths = [
            "clean_products125k-yolo",
            "products125k_generated_copy_paste",
            # "pure_background",
            "products125k_generated_bg_single",
            "products125k_hands_bg",
            "single_product_hands",
            "nondiff_multi_dense",
            "SKU-110K-r-yolo/train"
        ]
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            torch.manual_seed(worker_seed)
            torch.cuda.manual_seed(worker_seed)


        # train_size = 200
        # test_size = 100

        # # Ensure the dataset is large enough
        # if len(dataset) < train_size + test_size:
        #     raise ValueError(f"Dataset size {len(dataset)} is smaller than the required train size ({train_size}) + test size ({test_size}).")

        # # Split the dataset into training and test sets
        # train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, len(dataset) - train_size - test_size])

        
        paths = [Path(f"../data/{path}") for path in paths]
        # Remove DistributedSampler from dataset initialization
        dataset = ImageLabelDataset(root_dirs=paths, transform=train_transforms)
        
        print(f"Size of the dataset {len(dataset)}")

        train_split_ratio = 0.7
        train_size = int(train_split_ratio * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(42)  # Or any other fixed seed value
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        print(f"Train set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        # Use DistributedSampler in DataLoader if needed for distributed training
        train_sampler = DistributedSampler(train_dataset)  # Add this if using distributed training
        # test_sampler = DistributedSampler(test_dataset)    # Add this if needed for distributed testing

        # Create the DataLoader for both training and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, sampler=test_sampler, worker_init_fn=seed_worker)
        test_loader = None


    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return train_loader, test_loader
