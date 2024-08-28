import torch 
from einops import rearrange, einsum 
from utils.loss import _symmetric_kl, calculate_positive_loss, _calculate_outside_loss, _asymmetric_kl
import math
from collections import deque
import itertools
from utils.dist import init_loc, get_gaussian_map
from utils.smooth import GaussianSmoothing
import torch.nn.functional as F
smoother = GaussianSmoothing(1, 3, 0.5, 2).cuda()
def smooth(attn):
    _, P, t = attn.shape
    w = h = int(math.sqrt(P))
    attn_map = rearrange(attn, 'n (h w) c -> n c h w', h=h, w=w)
    smoothed = list()
    for i in range(t):
        input = attn_map[:, i].unsqueeze(dim=1)
        input = F.pad(input, (1, 1, 1, 1), mode='reflect')
        out = smoother(input)#.squeeze(0)#.squeeze(0)
        smoothed.append(out) 
    smoothed = torch.concat(smoothed, dim=1) 
    smoothed = rearrange(smoothed, 'n c h w -> n (h w) c')
    return smoothed

def TV_loss(x, weight=1):
    h, w = x.size()
    tv_h = torch.abs(x[1:,:] - x[:-1,:]).sum()
    tv_w = torch.abs(x[:,1:] - x[:,:-1]).sum()
    return weight * (tv_h + tv_w) / (h * w)




 
def auto_attn_gradient(prev_attn, ns, adjs, preserve_priors, eps=1e-5, 
                       preserve_scale=1., lse_scale=1., align_scale=1., disalign_scale=1., ablation=False):
    
    '''
        duplicate attn to avoid changing the cache
    ''' 

    if not ablation:
        attn_map = prev_attn.detach().clone() 
    else:
        attn_map = prev_attn


    '''
        enable auto grad
    '''

    with torch.enable_grad():
        attn_map = attn_map.requires_grad_(True)
        ori_attn_map = attn_map
        
        # attn_map = attn_map + eps

        # ori_attn.zero_grad()
        '''
            0. avg along and and reshape
        '''
        h = w = int(math.sqrt(attn_map.shape[1]))
        # attn_map = rearrange(ori_attn_map, 'n (h w) k -> n k h w', h=h, w=w)
        attn_map = attn_map.mean(dim=0, keepdim=True) 
        
        # attn_map = smooth(attn_map)
        attn_map = attn_map.squeeze(dim=0)
        attn_map = rearrange(attn_map, '(h w) k -> k h w', h=h, w=w)
        attn_map += eps

        '''
            1. alignment loss
        '''
        alignment_loss = 0.
        if len(adjs) != 0:
            assert len(adjs) == len(ns)

            for src, tgt in zip(adjs, ns):
                alignment_loss += calculate_positive_loss(attn_map, src, tgt, JS_div=False)

        if len(ns) != 0:
            alignment_loss /= len(ns)


        '''
            2. disalignment loss
        '''
        disalignment_loss = 0.
        for i in range(len(ns)):
            comb = deque(ns)
            comb.rotate(i)
            comb = list(comb) 
            src = comb[0]
            tgt = comb[1:]
            neg_loss, _ = _calculate_outside_loss(attn_map, src, tgt)
             
            if len(neg_loss) > 0:
                disalignment_loss -= sum(neg_loss) / len(tgt)
        # if len(ns) != 0:
        #     disalignment_loss /= len(ns)


        '''
            3. lse loss 
        '''

        lse_loss = 0.
        for n in ns:
            lse_loss = -1 * torch.logsumexp(attn_map[n].flatten(), dim=-1)

        if len(ns) != 0:
            lse_loss /= len(ns)


        '''
            3.1 balancing loss 
        '''
        bl_loss = 0. 
        smoothed_attn = smooth(rearrange(attn_map,'c h w -> 1 (h w) c')).squeeze(dim=0)
        bl_loss = 1 - smoothed_attn[:, list(ns)].max(dim=0).values
        bl_loss = bl_loss.mean()
        # bl_loss = attn_map[list(ns)]
        # bl_loss = 1 - rearrange(bl_loss, 'c h w -> c (h w)').max(dim=0).values.mean()



        '''
            4. area loss
        ''' 
        area_loss = 0.
        pairs = list(itertools.combinations(ns, 2))
        for src, tgt in pairs:
            area_loss += torch.nn.functional.mse_loss(attn_map[src], attn_map[tgt])
        if len(pairs) != 0:
            area_loss /= len(pairs)

        
        '''
            5. positional prior loss
        '''
        preserve_loss = 0.
        if preserve_priors is not None:
            # priors = priors.requires_grad_(True) 
            
            for n_idx, n in enumerate(ns):
                if n_idx >= len(preserve_priors):
                    continue

                prior = preserve_priors[n_idx].unsqueeze(dim=0).unsqueeze(dim=0)
                prior = torch.nn.functional.interpolate(prior, attn_map[n].shape)
                prior = prior.squeeze(dim=0).squeeze(dim=0)
                preserve_loss += _symmetric_kl(attn_map[n], prior)
                for other_n in ns:
                    if other_n != n:
                        preserve_loss -= _symmetric_kl(attn_map[other_n], prior)




        '''
            TV loss
        '''
        tv_loss = 0.
        for n in ns:
            tv_loss -= TV_loss(attn_map[n])
    
        
        '''
            compute gradint of d loss / d prev_attn

        '''
        loss = 0.
        loss += preserve_scale * preserve_loss
        loss += disalign_scale * disalignment_loss 
        loss += align_scale * alignment_loss  
        loss += lse_scale * bl_loss 
        grad_cond = torch.autograd.grad(
                loss.requires_grad_(True), [ori_attn_map], retain_graph=True, allow_unused=True
        )[0]
        
    if ablation:
        return loss
    
    return grad_cond

def dadq(J, K, idx=None):
    if idx is None:
        J_i = J
    else:
        J_i = J[:, :, idx]
    da_dq = einsum(J_i, K, 'n p z k, n k c -> n p z c')
    da_dq = rearrange(da_dq, 'n p z c -> z n p c')


    return da_dq 



def hidden_state_gradient(attn, ns, adjs, K, attn_scale, to_q_weight, preserve_prior, return_attn_grad=False,
                          preserve_scale=1, lse_scale=1., align_scale=1., disalign_scale=1.):
    if attn is None:
        return 0.
    attn = smooth(attn)
    attn_grad = auto_attn_gradient(attn, ns, adjs, preserve_prior,
                                   preserve_scale=preserve_scale, 
                                   lse_scale=lse_scale,
                                   align_scale=align_scale, 
                                   disalign_scale=disalign_scale)
    attn_grad = rearrange(attn_grad, 'n p k -> k n p 1')
    # d loss / d q 

    attn = attn.detach()
    attn.requires_grad_(False)
    with torch.no_grad():
        attn_prime = attn + torch.empty_like(attn).normal_(mean=0, std=0.01)
        # attn_prime = torch.clamp(attn_prime, 0.01, 0.99)
        J = torch.diag_embed(attn_prime) - torch.einsum ('npic, npjc -> npij', attn_prime.unsqueeze(dim=-1), attn_prime.unsqueeze(dim=-1))

        # tokens = list(ns + adjs)
        # grad = attn_grad[tokens] * dadq(J, K, tokens)
        grad = dadq(J, K) * attn_grad
        
        grad = rearrange(grad, 'z n p c -> z p (n c)')
        grad = einsum(grad, to_q_weight, 'z p c, c c1 -> z p c1')

        grad = grad * attn_scale
        grad = grad.sum(dim=0)
        grad = torch.stack([torch.zeros_like(grad), grad])

    if return_attn_grad:
        return grad, attn_grad
    return grad
