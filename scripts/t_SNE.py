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

# 🔹 1. Load Model & Data
device = "cuda" if torch.cuda.is_available() else "mps"
model = ResNet_Model(name="resnet34").to(device)
model.load_state_dict(torch.load("Loss_cos/Models/Epoch_100.pth", map_location=device))
model.eval()

# 🔹 2. Load CIFAR-100 Test Data
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.49, 0.48, 0.44], std=[0.24, 0.24, 0.26])])
test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 🔹 3. Extract Embeddings for Test Samples
all_embeds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        embeds = model(images).cpu().numpy()  # Get embeddings
        all_embeds.append(embeds)
        all_labels.extend(labels.numpy())

all_embeds = np.vstack(all_embeds)  # Shape: (N, 768)
subset_idx = np.random.choice(len(all_embeds), 5000, replace=False)  # Pick random points
all_embeds = all_embeds[subset_idx]
print(all_embeds.mean())


all_labels = np.array(all_labels)
all_labels = all_labels[subset_idx]


# 🔹 4. Load & Normalize Label Token Embeddings
label_tokens = np.load("../dream-ood/token_embed_c100.npy")  # Shape: (100, 768)
label_tokens = label_tokens/np.linalg.norm(label_tokens, axis=1, keepdims=True)
label_embeddings = label_tokens[all_labels]
print(label_tokens.mean())
# 🔹 5. Concatenate embeddings and label tokens
combined_embeds = np.vstack([all_embeds, label_tokens])  # Shape: (N + 100, 768)

# 🔹 6. Perform t-SNE once on the Embeddings
print('Fitting TSNE')
s = time.time()
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
all_tsne = tsne.fit_transform(combined_embeds)  # Shape: (N + 100, 2)
sample_tsne = all_tsne[:len(all_embeds)]
label_tsne  = all_tsne[len(all_embeds):]
print(f'Model Fitted in {time.time()-s}')



# 🔹 8. Plot the t-SNE Visualization
plt.figure(figsize=(8, 6))


plt.scatter(sample_tsne[:, 0], sample_tsne[:, 1], c=all_labels, cmap="tab20", alpha=0.5)

plt.scatter(label_tsne[:, 0], label_tsne[:, 1], c="black", marker="o", s=10, label="Label Tokens")

plt.legend()
plt.title("t-SNE of CIFAR-100 Sample Embeddings & Normalized Label Tokens")
plt.grid(True)
plt.savefig("../Images/test_tsne_plot.png")
plt.show()
