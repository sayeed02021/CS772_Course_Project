import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from natsort import natsorted
import glob 
import os
from sklearn.mixture import GaussianMixture
import math
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal



class GMM:
    def __init__(self, anchor_vec=None):
        self.mean = None
        self.cov = None
        # if anchor_vec is not None:
        #     self.anchor_vec_norm = anchor_vec.norm()
        # else:
        #     self.anchor_vec_norm = 1.0


    def fit(self, samples, anchor_vec=None):
        """Fits a GMM with 1 cluster.
        samples: NxD (normalized first)
        
        """
        if anchor_vec is not None:
            self.anchor_vec_norm = anchor_vec.norm() # update anchor vec
        samples = F.normalize(samples, dim=1) # *self.anchor_vec_norm
        # self.mean = torch.mean(samples, dim=0)# (D)
        # self.mean = F.normalize(self.mean, dim=0)
        # self.cov = torch.eye(samples.shape[1])*1e-2 # (DxD)
        # self.distr = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        self.gm = GaussianMixture(n_components=1, covariance_type='spherical', random_state=0)
        self.gm.fit(samples.numpy())


        self.mean = torch.tensor(self.gm.means_).squeeze()
        covar = self.gm.covariances_.item()
        self.exponent = 10**(round(math.log10(abs(covar))))
        # self.cov = self.exponent*torch.eye(samples.shape[1])
        self.cov = torch.eye(samples.shape[1])
        
        # print(self.exponent)
        self.distr = MultivariateNormal(self.mean, covariance_matrix=self.cov)



    def pdf(self, samples):
        """Calculates the PDF."""
        if self.mean is None or self.cov is None:
            raise ValueError("GMM must be fitted before calculating PDF.")

        samples = F.normalize(samples, dim=1) # *self.anchor_vec_norm
        log_pdf = self.distr.log_prob(samples)
        return torch.exp(log_pdf)

    def log_pdf(self, samples):
        """Calculates the log PDF."""
        if self.mean == None:
            raise ValueError("GMM must be fitted before calculating PDF.")
        # pdf_values = pdf_values.sum(axis=1)
        samples = F.normalize(samples, dim=1) # *self.anchor_vec_norm
        log_prob = self.distr.log_prob(samples)
        return log_prob

    def sgld_sampling(self, num_samples, eta=1e-3, initial_sample=None, burn=1000,n_chains=1):
        if self.mean ==None or self.cov==None:
            raise ValueError("GMM must be fitted before sampling.")

        chain_samples = num_samples//n_chains
        samples = []
        x = self.mean.clone()
        mean = self.mean.clone().detach()
        cov = self.cov.clone().detach()
        dist = MultivariateNormal(mean, covariance_matrix=cov)
        # x.requires_grad_(True)
        for chain in range(n_chains):
            x = mean + torch.randn_like(mean)
            x.requires_grad_(True)
            if chain==n_chains-1:
                chain_samples += num_samples%n_chains
            for iters in (range(chain_samples+burn)):
                logp = dist.log_prob(x)
                logp.backward()
                with torch.no_grad():
                    noise = torch.randn_like(x.detach())
                    x= x + 0.5*eta*x.grad + torch.sqrt(torch.tensor(eta)) * noise
                    if x.grad is not None:
                        x.grad.zero_()
                    if iters>=burn:
                        samples.append(x.clone().detach())
                x.requires_grad_(True)

        return torch.stack(samples)
    
    def reverse_sgld_sampling(self, num_samples, eta=1e-3, initial_sample=None, n_chains=1, burn=2000):
        """Performs outlier sampling through SGLD."""
        if self.mean ==None or self.cov==None:
            raise ValueError("GMM must be fitted before sampling.")
        chain_samples = num_samples//n_chains
        samples = []
        x = self.mean.clone()
        mean = self.mean.clone().detach()
        cov = self.cov.clone().detach()
        dist = MultivariateNormal(mean, covariance_matrix=cov)
        for chain in range(n_chains):
            x  = mean + torch.rand_like(mean)*eta
            x.requires_grad_(True)
            if chain==n_chains-1:
                chain_samples += num_samples%n_chains
            for iters in (range(chain_samples+burn)):
                logp = dist.log_prob(x)
                logp.backward()

                with torch.no_grad():
                    noise = torch.randn_like(x)
                    x_grad = x.grad
                    # print(x_grad)
                    # x_grad_scaled = torch.sqrt(torch.tensor(n_chains, dtype=torch.float32)) * x_grad
                    x_grad_scaled = x_grad
                    G = torch.exp(-2 * (x - self.mean).abs()) * torch.sqrt(torch.tensor(n_chains))*eta/2
                    # G = torch.exp(-2* (x - self.mean).norm()/self.mean.norm()) *eta/2
                    G2 = eta
                    x = x - G*x_grad_scaled +  torch.sqrt(torch.tensor(2 * eta)) * noise
                    # x = F.normalize(x, dim=0)
                    # print(G2)
                    if x.grad is not None:
                        x.grad.zero_()
                    if iters>=burn:
                        a = x.clone().detach()
                        samples.append(a)
                x.requires_grad_(True)
                

        return torch.stack(samples)
        



def test():





    torch.manual_seed(0)

    N,D = 500,2

    mean_true = torch.tensor([0.0, 0.0])
    cov_true =torch.eye(2)
    dist_true = torch.distributions.MultivariateNormal(mean_true, covariance_matrix=cov_true)
    samples = dist_true.sample((N,))  # shape: (N, 2)

    gmm = GMM()
    gmm.fit(samples)

    generated = gmm.sgld_sampling(num_samples=400, eta=0.1, burn=200, n_chains=5)
    generated = generated.cpu().detach()

    generated2 = gmm.reverse_sgld_sampling(num_samples=400, eta=0.1, burn=200, n_chains=5)
    generated2 = generated2.cpu().detach()

    samples = samples.cpu().detach()
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], label="Original samples", alpha=0.5)
    plt.scatter(generated[:, 0], generated[:, 1], label="SGLD samples", alpha=0.5)
    plt.scatter(generated2[:, 0], generated2[:, 1], label=" OOD SGLD samples", alpha=0.5)
    plt.legend()
    plt.title("GMM Fit and SGLD Sampling")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig('2D_sgld_results.png')
    plt.show()



def main():
    num_classes = 100

    path = 'Loss_cos'
    path2 = f'{path}/CIFAR100_train'
    files = glob.glob(path2+'/*.npy')
    files = natsorted(files)
    anchor_vecs = torch.tensor(np.load('token_embed_c100.npy'))

    model = GMM()
    save_path = f'{path}/CIFAR100_OOD'
    os.makedirs(save_path, exist_ok=True)
    for class_idx,file in tqdm(enumerate(files), total=len(files)):
        file_name = file.split('/')[-1]
        samples = torch.tensor(np.load(file))
        # anchor = anchor_vecs[class_idx, :]
        model.fit(samples)
        # print(model.mean, model.cov)

        gen_samples = model.reverse_sgld_sampling(num_samples=500, eta=5e-8, n_chains=1, burn=2500)
        gen_samples = gen_samples.cpu().detach().numpy()

        s_path = f'{save_path}/{file_name}'
        np.save(s_path, gen_samples)

    
if __name__=='__main__':
    test()
    # main()