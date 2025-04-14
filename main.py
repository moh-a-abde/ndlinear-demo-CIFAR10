import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.baseline import BaselineCNN
from models.ndlinear_model import NdLinearCNN
from utils.train import train_and_evaluate
from utils.visualize import create_plots
from utils.report import generate_markdown_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

baseline = BaselineCNN().to(device)
ndlinear = NdLinearCNN().to(device)

results = train_and_evaluate(baseline, ndlinear, train_loader, test_loader, device, epochs=30)
baseline_metrics, ndlinear_metrics, base_params, nd_params, bl_report, nd_report = results

create_plots(range(1, 31), baseline_metrics, ndlinear_metrics, base_params, nd_params)
generate_markdown_report(baseline_metrics, ndlinear_metrics, base_params, nd_params, bl_report, nd_report)
