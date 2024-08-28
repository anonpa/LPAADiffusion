import os 
import numpy as np 
from einops import rearrange
from sqlalchemy import all_
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

def save_attn(cros, dest, timestep, head, layer, name, size=256, normalize=False, div_sum=False):
    image = cros.numpy()[:, :]

    dest_dir = os.path.join(dest, timestep, head, layer)
    os.makedirs(dest_dir, exist_ok=True)

    if normalize:
        image = (image - image.min()) / (image.max() - image.min())

    elif div_sum:
        image = image / image.sum()
    image = 255 * image 
    image = image.astype(np.uint8)
    image = Image.fromarray(image).resize((size, size))
    image.save(os.path.join(dest_dir, name))

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
    "--raw",
    action='store_true',
)
parser.add_argument(
    "--div_sum",
    action='store_true',
)
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
    default='attn_vis',
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
    "--avg_head",
    action='store_true',
)
parser.add_argument(
    "--avg_layers",
    action='store_true',
)
parser.add_argument(
    "--avg_time",
    action='store_true',
)
parser.add_argument(
    "--KL_test",
    action='store_true',
)






args = parser.parse_args()
src = args.src
time_step = os.listdir(src) 
time_step.sort()
src_dir = [(t, os.path.join(src, t)) for t in time_step]
dest = args.dest
# to avoid duplicaiton
if os.path.exists(dest):
    shutil.rmtree(dest)
os.makedirs(dest, exist_ok=True)



attn_stat = defaultdict(list) 
all_attn = list()
for time_step, dir_path in track(src_dir):
    os.makedirs(os.path.join(dest, time_step), exist_ok=True)
    tmp = os.listdir(dir_path) 
    tmp.sort()
    # print(tmp)

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
    cros_attn_img = rearrange(cros_attn_img, 'l n k h w -> k n l h w')
    all_attn.append(cros_attn_img)
    
all_attn = torch.stack(all_attn, dim=0) # shape timestep, token, head, layer, h w


if args.avg_layers:
    all_attn = all_attn.mean(dim=3, keepdim=True) 

if args.avg_time:
    all_attn = all_attn.mean(dim=0, keepdim=True)

if args.avg_head:
    all_attn = all_attn.mean(dim=2, keepdim=True) 


n_timestep, n_token, n_head, n_layer, h, w = all_attn.shape
seleced_tokens = args.token
for t in track(range(n_timestep)):
    for k in range(n_token):
        if seleced_tokens is None or k in seleced_tokens:
            for h in range(n_head):
                for l in range(n_layer):
                    save_attn(all_attn[t, k, h, l], dest, str(t), str(h), str(l), f"{k}.png", normalize=args.normalize)


# compute mse 
all_attn = all_attn.mean(dim=2).mean(dim=2)[:, 2]
from collections import defaultdict
diff = defaultdict(list)

for i in range(n_timestep):
    for j in range(n_timestep):
        mse =((all_attn[i] - all_attn[j])**2).sum()
        idx = abs(i-j)
        diff[idx].append(mse)

mean_diff = list()
for idx in range(n_timestep):
    mean_diff.append(sum(diff[idx])/len(diff[idx]))
    
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(40,8))
plt.plot(mean_diff)
plt.xlabel('Timestep Difference')
plt.ylabel('MSE')
plt.savefig('line.png')



    


