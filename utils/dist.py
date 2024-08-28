import numpy as np
import torch 
from einops import rearrange, einsum
import random
import torch.distributions as dist


def adjust_distribution(distribution, index, target_value):
    # Total probability to redistribute

    # Exclude the element being increased (index 2, value 0.3)
    total_excluding_target = distribution.sum(dim=-1) - distribution[:, index]

    diff = target_value - distribution[:, index]

    # Calculate the proportional reduction factor
    reduction_factor = (total_excluding_target - diff) / total_excluding_target


    new_distribution = distribution * reduction_factor.unsqueeze(dim=-1)
    
    new_distribution[:, index] = target_value

    return new_distribution
def align_nouns(noun_phrase, nouns):

    anchor_idxs = list() 
    target_idxs = list()
    for np, n in zip(noun_phrase, nouns):
        np_txt, (np_s, np_e) = np 
        # randomly choice one noun in case of having multiple noun
        n_txt, (n_s, n_e) = random.choice(n) 
        
        np_s = np_s + 1 
        n_s = n_s + 1

        anchor_idxs.append(n_e) 
        target_idxs.append(list(range(np_s, n_e)) + list(range(n_e+1, np_e)))

    return anchor_idxs, target_idxs

         


def make_grid(x, y):
    # make a 2d grid 
    xs = torch.linspace(0, 1, steps=x)
    ys = torch.linspace(0, 1, steps=y)
    x, y = torch.meshgrid(xs, ys, indexing='xy') 
    grid = torch.stack([x, y], dim=-1)
    return grid



def get_multivariate_normal(samples):
    # sample shape: n c 
    mean = torch.mean(samples, dim=0).float()
    cov = torch.cov(samples.transpose(0, 1)).float()
    p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov)
    return p


def compute_log_prob(anchor, samples):
    p = get_multivariate_normal(anchor) 
    log_prob = p.log_prob(samples)
    return log_prob


def compute_cdf(anchor, samples):
    p = get_multivariate_normal(anchor) 
    cdf = p.cdf(samples)
    return cdf


def kl_div(p, q, samples):
    kl = p.log_prob(samples).exp() * (p.log_prob(samples) - q.log_prob(samples))
    return kl.sum()


def bhattacharyya(anchor, samples):
    mu_p = anchor.mean(dim=0).float() 
    cov_p = torch.cov(anchor.transpose(0, 1)).float()
    mu_q = samples.mean(dim=0).float() 
    cov_q = torch.cov(samples.transpose(0, 1)).float()

    delta_mu = mu_p - mu_q
    delta_mu = rearrange(delta_mu, 'c -> 1 c')
    sigma = (cov_p + cov_q) / 2

    a = einsum(delta_mu, torch.inverse(sigma), 'n c1, c1 c2 -> n c2')
    a = einsum(a, delta_mu, 'n c1, b c1 -> n b')

    b = 0.5 * torch.log10(torch.det(sigma) / torch.sqrt(torch.det(cov_p)*torch.det(cov_q)))
    dist =  a/8 + b
    return dist



'''
create 2d gaussian for given locs
'''

def get_gaussian_map(loc, cov=None, width=16, height=16, normalize=False, clamp=False):

    grid = make_grid(width, height)
    
    if loc is None:
        loc = [.5, .5]
    loc = torch.tensor(loc)  # Mean vector [mean_x, mean_y]
    if cov is None:
        cov = [[.01, 0.], [0., .01]]
    cov = torch.tensor(cov)  # Covariance matrix

    gaussian_2d = dist.MultivariateNormal(loc, cov)
    pdf = gaussian_2d.log_prob(grid).exp()

    if normalize:
        pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())
    
    if clamp:
        pdf = torch.clamp(pdf, 1e-3, 1)

    return pdf

def init_loc(n):

    return [[0.25, 0.5,], [0.75, 0.75]]

if __name__ == '__main__':
    pdf = get_gaussian_map([0.25, 0.5], None, 16, 16, normalize=True)
    pdf = pdf.numpy() * 255 
    pdf = pdf.astype(np.uint8)
    from PIL import Image 
    img = Image.fromarray(pdf) 
    img = img.resize((256, 256))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save('./gaussian.png')


