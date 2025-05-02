import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


from datasets import CIFAR100
from resent_model import ResNet_Model

dset = 'cifar100'
model_name = 'resnet34'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('mps')

params = torch.load('Loss_Cos/Models/Epoch_100.pth')
batch_size = 256


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

if dset.lower() == 'cifar100':
    test_data = CIFAR100(train=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False, pin_memory=False if device=='cpu' else True)
else:
    raise ValueError("Dataset not supported")


model = ResNet_Model(name = model_name).to(device)


model.load_state_dict(params)
print('All Parameters Matched')
model.eval()
fname = 'token_embed_c100.npy'
anchor = torch.from_numpy(np.load(fname, allow_pickle=True)).to(device) # shape: (100,768)
num_classes = 100
def validate():
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(device)
            x = model(image)
            out = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x), 1, 1),
                                x.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1 # output shape: batch_size x num_classes
            preds = out.argmax(dim=1).cpu()
            correct += (preds==label).sum().item()
            total  += len(label)


        print("Accuracy: ", correct/total)


if __name__=='__main__':
    validate()
    