import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from resent_model import ResNet_Model  # Import your ResNet model
from tqdm import tqdm
import time
import os
num_classes=100

device = "cuda" if torch.cuda.is_available() else "mps"
model = ResNet_Model(name="resnet34").to(device)
model.load_state_dict(torch.load("Loss_cos/Models/Epoch_100_loss_5.46.pth", map_location=device))
model.eval()


transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.49, 0.48, 0.44], std=[0.24, 0.24, 0.26])])
test_dataset = CIFAR100(root="../data", train=True, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


all_embeds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        embeds = model(images).cpu().numpy() # Get embeddings (Bx768)
        for i in range(embeds.shape[0]):
            all_embeds.append(embeds[i,:])
        all_labels.extend(labels.numpy())

all_embeds = np.array(all_embeds)
all_labels = np.array(all_labels)
print(all_embeds.shape, all_labels.shape)
os.makedirs('Loss_cos/CIFAR100_train', exist_ok=True)
for i in range(num_classes):
    idx = np.where(all_labels==i)[0]
    class_embeds = all_embeds[idx,:]
    class_name = f'../Loss_cos/CIFAR100_train/Class_{i}_test_embeds.npy'
    np.save(class_name, class_embeds)