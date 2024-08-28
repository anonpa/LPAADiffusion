import time
import argparse
import os
import math
import torch
from lpaa_diffusion_pipeline import LPAADiffusionPipeline
import time
import re
import shutil
import pandas as pd

def main(prompt, seed, output_directory, model_path, step_size, attn_res, lpaa_config=None, vis_attn=False, device='cpu', file=None, show_date=False, csv_path='./result.csv'):
    pipe = load_model(model_path, device=device)
    if os.path.exists('.attn_map'):
        shutil.rmtree('.attn_map')

    prompts = list() 
    if file is None:
        prompts.append(prompt)
    else:
        with open(file, 'r') as f:
            prompts = f.read().splitlines() 
            

    f_names = list()
    save_prompts = list()
    for idx, single_prompt in enumerate(prompts):
        single_prompt = re.sub('\s+',' ',single_prompt)
        image = generate(pipe, single_prompt, seed, step_size, attn_res, vis_attn, device=device, lpaa_config=lpaa_config)
        file_name = save_image(image, str(idx), seed, output_directory, show_date=show_date)
        f_names.append(file_name) 
        save_prompts.append(single_prompt)

    data = dict(filename=f_names, prompts=save_prompts)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, sep='\t', index=False)


def load_model(model_path, device='cpu'):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(device)
    pipe = LPAADiffusionPipeline.from_pretrained(model_path).to(device)
    return pipe


def generate(pipe, prompt, seed, step_size, attn_res, vis_attn, device='cpu', lpaa_config=dict(scale=100, prior_scale=10, lse_scale=1, align_scale=1, area_scale=1, disalign_scale=1, beta=0.9), probar=True):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(device)
    generator = torch.Generator(device.type).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size, num_inference_steps=50, probar=probar, attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))), vis_attn=vis_attn, lpaa_config=lpaa_config)
    return result['images'][0]


def save_image(image, save_name, seed, output_directory, ext='png', show_date=False):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    date = str(time.time())
    
    appendix = ""
    if show_date:
        appendix = f'_{seed}_{date}'
    dir_name = f"{save_name}{appendix}.{ext}"
    file_name = f"{output_directory}/{dir_name}"
    image.save(file_name)


    return dir_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a checkered bowl on a red and blue table"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='./output'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=20.0,
        help='the step size for single pass, ablation only',
    )

    parser.add_argument(
        '--attn_res',
        type=int,
        default=256,
        help='The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)'
    )

    parser.add_argument(
        '--vis_attn',
        action='store_true',
        help='save attention map for visualization',
    )
    parser.add_argument(
        '--show_date',
        action='store_true',
        help='add date to image name, to avoid overwriting',
    )
    parser.add_argument(
        '--include_entities',
        action='store_true',
    )

    parser.add_argument(
        '--num',
        type=int,
        default=None
    )

    # 
    parser.add_argument(
        "--scale",
        type=float,
        default=100.,
    )
    parser.add_argument(
        "--preserve_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--lse_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--align_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--area_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--disalign_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--mock",
        action="store_true"
    )
    parser.add_argument(
        "--single_pass",
        action="store_true"
    )
    parser.add_argument(
        "--target_res",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--target_layers",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    lpaa_config = dict(
                scale=args.scale,
                preserve_scale =args.preserve_scale,
                lse_scale = args.lse_scale,
                align_scale = args.align_scale,
                area_scale = args.area_scale,
                disalign_scale = args.disalign_scale,
                single_pass = args.single_pass,
                cache=args.cache,
                beta = args.beta,
                target_res = args.target_res,
                target_layers = args.target_layers
            )
    
    main(args.prompt, args.seed, args.output_directory, args.model_path, args.step_size, args.attn_res,
         vis_attn=args.vis_attn, device=args.device, file=args.file, show_date=args.show_date, lpaa_config=lpaa_config)
