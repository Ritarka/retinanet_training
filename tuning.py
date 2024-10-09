import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial
import os
import matplotlib.pyplot as plt

import gc
gc.collect()


# Specify font family to avoid font warnings on Colab
plt.rcParams['font.family'] = 'DejaVu Sans'

torch.cuda.empty_cache()

# In[2]:
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from data import get_dataloaders
import neural_network
import trainer
import visualize

torch.cuda.memory_summary(device=None, abbreviated=False)
# Set up distributed training
dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.getenv("LOCAL_RANK", 0))  # Use LOCAL_RANK from distributed environment
device = f'cuda:{local_rank}'
torch.cuda.set_device(local_rank)

# Hyperparameters
batch_size = 1
learning_rate = 0.02
epochs = 40

# Data transforms
transform_train = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Get data loaders with DistributedSampler
trainloader, testloader = get_dataloaders(
    dataset_name="products",
    batch_size=batch_size,
    train_transforms=transform_train,
    test_transforms=transform_test,
)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# Initialize the model
model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            fg_iou_thresh = 0.85,
            bg_iou_thresh = 0.5
        )
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head = RetinaNetClassificationHead(
    in_channels=256,
    num_anchors=num_anchors,
    num_classes=2,
    norm_layer=partial(torch.nn.GroupNorm, 32)
)

# Freeze the backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Use RetinaNet's internal loss functions
class_head = [param for param in model.head.classification_head.parameters() if param.requires_grad]
reg_head = [param for param in model.head.regression_head.parameters() if param.requires_grad]

# Move model to device before wrapping it in DDP
model = model.to(device)

# Wrap the model in DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, 
                                                  #find_unused_parameters=True, 
                                                  )


optimizer_class = torch.optim.SGD(class_head, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer_reg = torch.optim.SGD(class_head, lr=learning_rate, momentum=0.9, weight_decay=5e-4)


criterion = nn.CrossEntropyLoss()

# Training loop
train_losses, test_losses = trainer.train(model, optimizer_class, optimizer_reg, criterion, trainloader, testloader, epochs, device)

# Evaluate and visualize results
# accuracy = trainer.evaluate_accuracy(model, testloader, device)
# print("Test Accuracy:", accuracy)
# visualize.visualize_loss(train_losses, test_losses, "Best Model")
# visualize.visualize_images(model, testloader.dataset, device, "Best Model")
