import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os
import glob
import time
from tqdm import tqdm


from resent_model import ResNet_Model
from datasets import CIFAR100

# overall settings
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else: 
    device = torch.device('cpu')
dset = 'cifar100'
model_name = 'resnet34'
num_epochs=100
print(num_epochs)
batch_size = 256
num_classes = 100
normalize = True

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

if dset.lower() == 'cifar100':
    root = '../data'
    train_data = CIFAR100(train=True,root=root, transform=train_transform)
    test_data = CIFAR100(train=False,root=root, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, pin_memory=False if device=='cpu' else True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False, pin_memory=False if device=='cpu' else True)
else:
    raise ValueError("Dataset not supported")


criterion = nn.CrossEntropyLoss()
# Define model parameters
print("Using Device: ", device)

model = ResNet_Model(name = model_name).to(device)
ngpu=0
if device!='cpu':
    ngpu = torch.cuda.device_count()
    print("Number of GPUs: ", ngpu) 

if ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, 
                      weight_decay = 5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs*len(train_loader), eta_min = 1e-6)

# Load Tokens
if ngpu<1:
    fname = 'token_embed_c100.npy'
else:
    fname = '/kaggle/input/scripts-for-running/token_embed_c100.npy'
print(fname)
anchor = torch.from_numpy(np.load(fname, allow_pickle=True)).to(device) # shape: (100,768)
# anchor = F.normalize(anchor, dim=1, )
# criterion  = ContrastiveLoss(tokens=anchor)



# Training Function
def training(epoch):

    model.train()
    avg_loss = 0
    # start  = time.time()
    pbar = tqdm(train_loader)
    for idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        x = model(images, normalize=normalize) # Shape: batch_size x 768
        out = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x), 1, 1),
                                x.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1 # output shape: batch_size x num_classes
        loss = criterion(out,labels)   # For CE Loss
        # loss = criterion(x, labels)
        loss.backward()
        optimizer.step()
        
        avg_loss +=loss.item()
        scheduler.step()
        stats = {

            'Epoch': epoch+1,
            'Avg_Loss': f'{avg_loss/(idx):0.3f}',
            'Lr': f'{scheduler.get_last_lr()[0]:0.2e}'
        }
        pbar.set_postfix(stats)


    # end = time.time()
    # avg_loss /=len(train_loader)
    # lr = scheduler.get_last_lr()[0]
    # print(f'Epoch: {epoch+1}, Loss: {avg_loss: 0.3f}, Time Taken: {end-start: 0.2f} sec, LR: {lr :0.6f}')
    return avg_loss

def main():
    os.makedirs('Loss_Cos/Models')
    print('Starting Training')
    for epoch in range(num_epochs):
        loss = training(epoch)
        if (epoch+1)%10 ==0:
            if ngpu>1:
                model_copy = model.module.cpu()
            else:
                model_copy = model.cpu()
            torch.save(model_copy.state_dict(), f'../Los_Cos/Models/Epoch_{epoch+1}.pth')
            model.to(device)
            print('Model Saved')



if __name__=='__main__':
    main()
