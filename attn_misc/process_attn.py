import os 
import numpy as np 
from einops import rearrange
import torch
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from rich.progress import track
import shutil
import argparse
import itertools
def process_self_attn(attn, res=64):
    attn = np.stack(attn, axis=0).mean(axis=0)

    return attn 

def get_stat(attn, show_stat=False):
    if show_stat:
        return f"_{attn.min():.2f}_{attn.max():.2f}_{attn.mean():.2f}"
    return ""

def save_attn(cros, dest, size=256, normalize=False, div_sum=False):
    image = cros.numpy()[:, :]

    # normalize 
    if normalize:
        image = (image - image.min()) / (image.max() - image.min())

    elif div_sum:
        image = image / image.sum()
    image = 255 * image 
    image = image.astype(np.uint8)
    image = Image.fromarray(image).resize((size, size))
    image.save(dest)

def process_cros_attn(attn, target_res=None, res=64):
    aa = list()
    for a in attn:
        h = a.shape[1] 
        h = int(h**0.5)
        if target_res is None or h in target_res:
            a = rearrange(a, 'b (h w) c -> b h w c', h=h, w=h)
            a = torch.from_numpy(a) 
            a = rearrange(a, 'b h w c -> b c h w')
            a = torch.nn.functional.interpolate(a.to(torch.float32), (res, res), mode='nearest')
            # if not per_head:
            #     a = a.max(dim=0)[0]
            # a = a.mean(dim=0)
            aa.append(a)

    # aa = torch.stack(aa).mean(dim=0)
    if len(aa) != 0:
        aa = torch.stack(aa)

    return aa
# 
parser = argparse.ArgumentParser()


parser.add_argument(
    "--normalize",
    action='store_true',
)

parser.add_argument(
    "--show_stat",
    action='store_true',
)

parser.add_argument(
    "--src",
    type=str,
    default='.attn_map',
)

parser.add_argument(
    "--dest",
    type=str,
    default='attn_data',
)

parser.add_argument(
    "--out_res",
    type=int,
    default=16,
)
parser.add_argument(
    "--res",
    nargs='+',
    type=int,
    default=None,
)
parser.add_argument(
    "--token",
    nargs='+',
    type=int,
    default=None,
)

parser.add_argument(
    "--avg_layers",
    action='store_true',
)
parser.add_argument(
    "--avg_heads",
    action='store_true',
)

parser.add_argument(
    "--KL_test",
    action='store_true',
)






args = parser.parse_args()
src = args.src
src_dir = [(time_step, os.path.join(src, time_step)) for time_step in os.listdir(src)]
dest = args.dest
# to avoid duplicaiton
if os.path.exists(dest):
    shutil.rmtree(dest)
os.makedirs(dest, exist_ok=True)



attn_stat = defaultdict(list) 


time_step_data = list()
for time_step, dir_path in track(src_dir):
    # os.makedirs(os.path.join(dest, time_step), exist_ok=True)
    tmp = os.listdir(dir_path) 
    tmp.sort()

    self_attn = list()
    cros_attn = list()
    for attn in tmp:
        attn_map = np.load(os.path.join(dir_path, attn)).squeeze()
        if attn_map.shape[-1] == 77:
            cros_attn.append(attn_map) 
        else:
            self_attn.append(attn_map)
    
    # self_attn_img = process_self_attn(self_attn)
    cros_attn_img = process_cros_attn(cros_attn, target_res=args.res, res=args.out_res)
    cros_attn_img = rearrange(cros_attn_img, 'l n k h w -> l n k (h w)')
    time_step_data.append(cros_attn_img)

time_step_data = torch.stack(time_step_data, dim=2)
print(time_step_data.shape)

if args.avg_layers:
    time_step_data = time_step_data.mean(dim=0, keepdim=True)

if args.avg_heads:
    time_step_data = time_step_data.mean(dim=1, keepdim=True)


# plot graph 
l, n, t, _, _ =  time_step_data.shape

for i in range(l):
    os.makedirs(os.path.join(dest, f"layer_{i}"))
    for j in range(n):
        # os.makedirs(os.path.join(dest, f"layer_{i}", f"head_{j}"))
        single_data = time_step_data[i, j].mean(dim=-1).cpu().numpy()
        single_data = rearrange(single_data, 't k -> k t')
        if args.token is not None:
            single_data = single_data[args.token]
        plt.clf()
        for token_idx, single_line in enumerate(single_data):
            plt.plot(single_line, label = f"{token_idx}") 
            plt.legend() 
        plt.savefig(os.path.join(dest, f"layer_{i}", f"head_{j}.png"))



    

    

